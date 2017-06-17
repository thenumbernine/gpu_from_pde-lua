#!/usr/bin/env luajit
require 'ext'
local template = require 'template'
local symmath = require 'symmath'
local gnuplot = require 'gnuplot'
local matrix = require 'matrix'
local CLEnv = require 'cl.obj.env'
local CLGMRES = require 'solver.cl.gmres'
local clnumber = require 'cl.obj.number'

local derivativeOrder = 2

local x, y = symmath.vars('x', 'y')
local coords = table{x, y}
symmath.Tensor.coords{
	{variables=coords},
	{symbols='x', variables={x}},
	{symbols='y', variables={y}},
}

local u = symmath.var('u', {x, y})

local alpha = symmath.var('alpha', nil, 10)

local params = table{alpha}

local pde = alpha * (u',xx' + u',yy')

pde = pde()
print(pde)

-- because :map is bottom up, I need to collect diffs and remove them first
local vars = table()
local diffvars = table()
pde = pde:map(function(expr)
	if symmath.diff.is(expr) then
		local name = table.map(expr, function(v) return v.name end):concat'_'
		local v = vars[name]
		if not v then
			v = symmath.var(name)
			diffvars:insert{[name]=expr}
			vars[name] = v
		end
		return v
	end
end):map(function(expr)
	if symmath.var.is(expr) then
		vars[expr.name] = expr
	end
end)
vars = vars:kvpairs()
local uCode = pde:compile(vars, 'C')

local env = CLEnv{size={256,256}}

local volume = tonumber(env.base.volume)

local xmin = {-1, -1}
local xmax = {1, 1}

env.code = env.code .. template([[
constant const real2 xmin = (real2)(<?=clnumber(xmin[1])?>, <?=clnumber(xmin[2])?>);
constant const real2 xmax = (real2)(<?=clnumber(xmax[1])?>, <?=clnumber(xmax[2])?>);
constant const real2 dx = (real2)(
	<?=clnumber((xmax[1] - xmin[1]) / tonumber(env.base.size.x))?>,
	<?=clnumber((xmax[2] - xmin[2]) / tonumber(env.base.size.y))?>);
#define cell_x(i) (xmin + ((real2)(i.x, i.y) + (real2)(.5, .5)) * dx)
#define OOB(i)	(i.x < 0 || i.y < 0 || i.x >= size.x || i.y >= size.y)
#define boundary(i)	0.
inline real calc_u<?=uCode?>

]], {
	env = env,
	xmin = xmin,
	xmax = xmax,
	clnumber = clnumber,
	uCode = uCode,
})

-- initialize u
local uBuf = env:buffer{name='u', type='real'}
uBuf:fill()

local u0Buf = env:buffer{name='u0', type='real'}
u0Buf:copyFrom(uBuf)

local rhoBuf = env:buffer{name='rho', type='real'}
env:kernel{
	argsOut={rhoBuf},
	body=[[
	real2 x = cell_x(i);
	rho[index] = dot(x,x) < .5 *.5 ? .1 : 0.;
]]
}()

-- derivCoeffs[derivative][accuracy] = {coeffs...}
local derivCoeffs = {
	-- antisymmetric coefficients 
	{
		[2] = {.5},
		[4] = {2/3, -1/12},
		[6] = {3/4, -3/20, 1/60},
		[8] = {4/5, -1/5, 4/105, -1/280},
	},
	-- symmetric
	{
		[2] = {[0] = -2, 1},
		[4] = {[0] = -5/2, 4/3, -1/12},
		[6] = {[0] = -49/18, 3/2, -3/20, 1/90},
		[8] = {[0] = -205/72, 8/5, -1/5, 8/315, -1/560},
	},
	-- antisymmetric 
	{
		[2] = {-1, 1/2},
		[4] = {-13/8, 1, -1/8},
		[6] = {-61/30, 169/120, -3/10, 7/240},
	},
	-- symmetric
	{
		[2] = {[0] = 6, -4, 1},
		[4] = {[0] = 28/3, -13/2, 2, -1/6},
		[6] = {[0] = 91/8, -122/15, 169/60, -2/5, 7/240},
	},
	-- antisymmetric 
	{
		[2] = {5/2, -2, 1/2},
	},
	-- symmetric
	{
		[2] = {[0] = -20, 15, -6, 1},
	},
}

-- now take the derivative variables and collect the source variables required for the finite derivative kernel
local function suffixForOffset(v) return table.concat(v, '_'):gsub('-','m') end
local offsets = table()
if vars:find(u) then offsets:insert(matrix{0,0}) end
local lines = table()
for _,kv in ipairs(diffvars) do
	local name, v = next(kv)
	-- count the # of unique variables
	local ds = matrix{0,0}
	for i=2,#v do
		local j = assert(coords:find(v[i]))
		ds[j] = ds[j] + 1
	end
	local numNonZero = 0
	local side
	for i=1,#ds do
		if ds[i] ~= 0 then 
			numNonZero = numNonZero + 1 
			side = i
		end
	end

	if numNonZero == 1 then
		local function ofs(i)
			local xp = matrix{2}:zeros() 
			xp[side] = i
			if not offsets:find(xp) then offsets:insert(xp) end
			xp = suffixForOffset(xp)
			return 'u_'..xp
		end
		
		-- numerical derivatives along a single axis
		local count = ds[side]
		local coeffs = derivCoeffs[count][derivativeOrder]

		local args = {
			name = name,
			ofs = ofs,
			side = side,
			num = range(-#coeffs,#coeffs):map(function(i) 
				local coeff = coeffs[math.abs(i)]
				return coeff and (clnumber(coeff)..' * '..ofs(i)) or '0.'
			end):concat' + ',
			dx = range(count):map(function() 
				return 'dx.s'..(side-1) 
			end):concat' * ',
		}
		if coeffs then
			lines:insert(template([[	real <?=name?> = (<?=num?>) / (<?=dx?>);]], args))
		elseif count == 2 then
			error("haven't got this derivative order yet?")
		end
	else
		error("haven't got support for mixed derivatives just yet")
	end
end

local ofslines = table()
for _,offset in ipairs(offsets) do
	ofslines:insert(template([[
	int4 i_<?=suffix?> = i;
	i_<?=suffix?>.x += <?=offset[1]?>;
	i_<?=suffix?>.y += <?=offset[2]?>;
	real u_<?=suffix?> = OOB(i_<?=suffix?>) ? boundary(i_<?=suffix?>) : u[indexForInt4(i_<?=suffix?>)];
]], 	{
			offset = offset,
			suffix = suffixForOffset(offset),
		}))
end
lines = ofslines:append(lines)

local paramlines = table()
for _,param in ipairs(params) do
	local name = param.name
	local value = param.value
	paramlines:insert(template([[	const real <?=name?> = <?=clnumber(value)?>;]], {
		clnumber = clnumber,
		name = name,
		value = value,
	}))
end
lines = paramlines:append(lines)

lines = lines:concat'\n'

local gmres = CLGMRES{
	env = env,
	A = env:kernel{
		argsOut = {{name='Au', type='real', obj=true}},
		argsIn = {{name='u', type='real', obj=true}},
		body = template([[
<?=lines?>
	Au[index] = calc_u(<?=vars:map(function(kv) return (next(kv)) end):concat', '?>);
]],		{
			vars = vars,
			clnumber = clnumber,
			lines = lines,
			alpha = .01,
		}),
	},
	b = rhoBuf,
	x = uBuf,
	epsilon = .01,
	restart = 30,
	errorCallback = function(err,iter)
		io.stderr:write(tostring(err)..'\t'..tostring(iter)..'\n')
		assert(err == err)
	end,
}
local olddot = gmres.args.dot
gmres.args.dot = function(...)
	return olddot(...) / volume
end
gmres()

local function gpuToTable(x)
	local cpu = x:toCPU()
	local t = table()
	for i=1,tonumber(env.base.size.x) do
		t[i] = table()
		for j=1,tonumber(env.base.size.y) do
			t[i][j] = cpu[(j-1) + env.base.size.x * (i-1)]
		end
	end
	return t
end

local xs = range(tonumber(env.base.size.x)):map(function(i) return (i-.5)*(xmax[1]-xmin[1])/tonumber(env.base.size.x) + xmin[1] end)
local ys = range(tonumber(env.base.size.y)):map(function(i) return (i-.5)*(xmax[2]-xmin[2])/tonumber(env.base.size.y) + xmin[2] end)

for _,info in ipairs{
	{u=uBuf}, 
	--{u0=u0Buf},
	{rho=rhoBuf},
} do
	local name, buf = next(info)
	gnuplot{
		output = name..'.png',
		style = 'data lines',
		griddata = {x=xs, y=ys, gpuToTable(buf)},
		{splot=true, using='1:2:3', title=name},
	}
end
