#!/usr/bin/env luajit
require 'ext'
local template = require 'template'
local clnumber = require 'cl.obj.number'
require 'symmath'.setup()

local x, y = vars('x', 'y')
local u = var'u'
local u_xx = var'u_xx'
local u_yy = var'u_yy'
local alpha = var'alpha'

local pde = alpha * (u_xx + u_yy)

local f = pde:compile({u, x, y, u_xx, u_yy, alpha}, 'C')

local env = require 'cl.obj.env'{size={256,256}}

local xmin = {-1, -1}
local xmax = {1, 1}

env.code = env.code .. template([[
constant const real2 xmin = (real2)(<?=clnumber(xmin[1])?>, <?=clnumber(xmin[2])?>);
constant const real2 xmax = (real2)(<?=clnumber(xmax[1])?>, <?=clnumber(xmax[2])?>);
constant const real2 dx = (real2)(
	<?=clnumber((xmax[1] - xmin[1]) / tonumber(env.base.size.x))?>,
	<?=clnumber((xmax[2] - xmin[2]) / tonumber(env.base.size.y))?>);
#define cell_x(i) (xmin + ((real2)(i.x, i.y) + (real2)(.5, .5)) * dx)

//boundary conditions
#define boundaryXP(i) 0.
#define boundaryXM(i) 0.
#define boundaryYP(i) 0.
#define boundaryYM(i) 0.
]], {
	env = env,
	xmin = xmin,
	xmax = xmax,
	clnumber = clnumber,
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
	rho[index] = dot(x,x) < .5*.5 ? .1 : 0.;
]]
}()

-- solve u,xx + u.yy = rho
require 'solver.cl.gmres'{
	env = env,
	A = env:kernel{
		argsOut = {{name='Au', type='real', obj=true}},
		argsIn = {{name='u', type='real', obj=true}},
		body = template([[
	const real alpha = <?=clnumber(alpha)?>;	
	
	real ui = u[index];
	int4 ixp = i;
	ixp.x++;
	real uxp = i.x < size.x-1 ? u[indexForInt4(ixp)] : boundaryXP(i);
	int4 ixm = i;
	ixm.x--;
	real uxm = i.x > 0 ? u[indexForInt4(ixm)] : boundaryXM(i);
	int4 iyp = i;
	iyp.y++;
	real uyp = i.y < size.y-1 ? u[indexForInt4(iyp)] : boundaryYP(i);
	int4 iym = i;
	iym.y--;
	real uym = i.y > 0 ? u[indexForInt4(iym)] : boundaryYM(i);

	const real u_xx = (uxp - 2 * ui + uxm) / (dx.x * dx.x);
	const real u_yy = (uyp - 2 * ui + uym) / (dx.y * dx.y);
	
	Au[index] = alpha * (u_xx + u_yy);
]],		{
			clnumber = clnumber,
			alpha = .01,
		}),
	},
	b = rhoBuf,
	x = uBuf,
	epsilon = 1,
	restart = 30,
	errorCallback = function(err,iter)
		io.stderr:write(tostring(err)..'\t'..tostring(iter)..'\n')
		assert(err == err)
	end,
}()

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

local gnuplot = require 'gnuplot'
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
