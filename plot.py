# MIT License

# Copyright (c) 2024 Kangyao Huang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import plotly.graph_objects as go
from math import comb
from scipy.interpolate import CubicSpline
from sympy import symbols, expand

def bernstein_poly(i, n, t):
    """
    计算贝塞尔基函数B_{i,n}(t)
    """
    return comb(n, i) * (t**i) * ((1 - t)**(n - i))

def bezier_curve_with_equations(control_points):
    """
    输出贝塞尔曲线的数学表达式，并计算其曲线坐标
    :param control_points: 控制点列表，形如 [(x1, y1, z1), (x2, y2, z2), ...]
    :return: 贝塞尔曲线的坐标列表 [(x, y, z), ...]
    """
    n = len(control_points) - 1
    t = symbols('t')
    
    # 打印贝塞尔曲线的数学表达式
    print("贝塞尔曲线的多项式表达式：")
    
    bezier_x = 0
    bezier_y = 0
    bezier_z = 0
    for i, (px, py, pz) in enumerate(control_points):
        # 贝塞尔基函数
        b = bernstein_poly(i, n, t)
        
        # 计算每个维度的多项式
        bezier_x += b * px
        bezier_y += b * py
        bezier_z += b * pz
    
    # 展开并打印每个维度的多项式表达式
    bezier_x_expanded = expand(bezier_x)
    bezier_y_expanded = expand(bezier_y)
    bezier_z_expanded = expand(bezier_z)

    print(f"x(t) = {bezier_x_expanded}")
    print(f"y(t) = {bezier_y_expanded}")
    print(f"z(t) = {bezier_z_expanded}")
    
    # 计算贝塞尔曲线上的坐标
    t_values = np.linspace(0, 1, 100)
    curve_points = []
    for t_val in t_values:
        x = sum(bernstein_poly(i, n, t_val) * control_points[i][0] for i in range(n+1))
        y = sum(bernstein_poly(i, n, t_val) * control_points[i][1] for i in range(n+1))
        z = sum(bernstein_poly(i, n, t_val) * control_points[i][2] for i in range(n+1))
        curve_points.append((x, y, z))
    
    return curve_points

def bezier_curve(control_points, t_values):
    """
    计算贝塞尔曲线
    :param control_points: 控制点列表，形如 [(x1, y1, z1), (x2, y2, z2), ...]
    :param t_values: t的取值列表，范围在 [0, 1] 之间
    :return: 计算得到的曲线坐标列表 [(x, y, z), ...]
    """
    n = len(control_points) - 1
    curve_points = []
    
    for t in t_values:
        x = y = z = 0
        for i, (px, py, pz) in enumerate(control_points):
            b = bernstein_poly(i, n, t)
            x += b * px
            y += b * py
            z += b * pz
        curve_points.append((x, y, z))
    
    bezier_curve_with_equations(control_points)
    return curve_points

def cubic_spline_curve(control_points, t_values):
    """
    使用 scipy 的 CubicSpline 计算三次样条曲线
    :param control_points: 控制点列表，形如 [(x1, y1, z1), (x2, y2, z2), ...]
    :param t_values: t的取值列表，范围在 [0, 1] 之间
    :return: 计算得到的曲线坐标列表 [(x, y, z), ...]
    """
    # 将控制点转化为数组
    control_points = np.array(control_points)
    # 生成参数 t，假设 t 与控制点数量一一对应
    t_knots = np.linspace(0, 1, len(control_points))
    
    # 对每个维度分别进行三次样条插值
    spline_x = CubicSpline(t_knots, control_points[:, 0], bc_type='natural')
    spline_y = CubicSpline(t_knots, control_points[:, 1], bc_type='natural')
    spline_z = CubicSpline(t_knots, control_points[:, 2], bc_type='natural')
    
    # 使用插值函数计算曲线上的点
    spline_x_vals = spline_x(t_values)
    spline_y_vals = spline_y(t_values)
    spline_z_vals = spline_z(t_values)

    # 打印每段曲线的数学表达式
    print("三次样条曲线的多项式表达式：")
    t = symbols('t')
    for i in range(len(t_knots) - 1):
        # 获取第i段的系数
        coefficients_x = spline_x.c[:, i]
        coefficients_y = spline_y.c[:, i]
        coefficients_z = spline_z.c[:, i]

        # 将系数保留到小数点后一位
        coefficients_x = [round(c, 1) for c in coefficients_x]
        coefficients_y = [round(c, 1) for c in coefficients_y]
        coefficients_z = [round(c, 1) for c in coefficients_z]
        
        # 构造每个维度的三次多项式表达式
        poly_x = sum([coefficients_x[j] * t**(3-j) for j in range(4)])
        poly_y = sum([coefficients_y[j] * t**(3-j) for j in range(4)])
        poly_z = sum([coefficients_z[j] * t**(3-j) for j in range(4)])
        
        # 打印每段的多项式
        print(f"\nSegment {i+1}:")
        print(f"x(t) = {expand(poly_x)}")
        print(f"y(t) = {expand(poly_y)}")
        print(f"z(t) = {expand(poly_z)}")
    
    # 返回三次样条曲线的点
    return list(zip(spline_x_vals, spline_y_vals, spline_z_vals))

# 示例控制点
control_points = [(0, 0, 0), (2, 2, 1), (4, 3, 2), (6, 0, 2), (13, 2, 1)]
x_coords, y_coords, z_coords = zip(*control_points)
# 计算每个轴的最小值和最大值
x_range = [min(x_coords), max(x_coords)]
y_range = [min(y_coords), max(y_coords)]
z_range = [min(z_coords), max(z_coords)]

# 生成贝塞尔曲线的点
t_values = np.linspace(0, 1, 100)
bezier_points = bezier_curve(control_points, t_values)

# 生成三次样条曲线的点
spline_points = cubic_spline_curve(control_points, t_values)

# 提取曲线的x, y, z坐标
bezier_x, bezier_y, bezier_z = zip(*bezier_points)
spline_x, spline_y, spline_z = zip(*spline_points)

# 使用 Plotly 绘制3D贝塞尔曲线和三次样条曲线
fig = go.Figure(data=[go.Scatter3d(x=bezier_x, y=bezier_y, z=bezier_z, mode='lines', name='Bezier Curve',
                                   line=dict(width=6, color='blue')),  # 贝塞尔曲线
                     go.Scatter3d(x=spline_x, y=spline_y, z=spline_z, mode='lines', name='Cubic Spline Curve',
                                   line=dict(width=6, color='green')),  # 三次样条曲线
                     go.Scatter3d(x=[p[0] for p in control_points], 
                                  y=[p[1] for p in control_points], 
                                  z=[p[2] for p in control_points], 
                                  mode='markers+text', text=["P" + str(i) for i in range(len(control_points))], 
                                  marker=dict(size=10, color='red'), name='Control Points')])

# 修改坐标轴样式
fig.update_layout(
    scene=dict(
        aspectmode='data',
        xaxis=dict(
            # tickmode='auto',
            title='X',
            titlefont=dict(size=14, family='Arial, sans-serif', color='black'),
            tickfont=dict(size=12, family='Arial, sans-serif', color='black'),
            showgrid=True,
            gridcolor='gray',
            zeroline=True,
            zerolinecolor='gray'
        ),
        yaxis=dict(
            # tickmode='auto',
            title='Y',
            titlefont=dict(size=14, family='Arial, sans-serif', color='black'),
            tickfont=dict(size=12, family='Arial, sans-serif', color='black'),
            showgrid=True,
            gridcolor='gray',
            zeroline=True,
            zerolinecolor='gray'
        ),
        zaxis=dict(
            # tickmode='auto',
            title='Z',
            titlefont=dict(size=14, family='Arial, sans-serif', color='black'),
            tickfont=dict(size=12, family='Arial, sans-serif', color='black'),
            showgrid=True,
            gridcolor='gray',
            zeroline=True,
            zerolinecolor='gray'
        ),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    title="3D Bezier Curve and Cubic Spline Curve Visualization",
    titlefont=dict(size=24, family='Arial, sans-serif', color='black'),
    showlegend=True
)

fig.show()


def bezier_equation_simplified(control_points):
    n = len(control_points) - 1
    equations = []
    
    # 简化贝塞尔曲线表达式，忽略组合数 C(n,i)
    for i in range(n + 1):
        # 对每个维度的表达式进行简化
        term_x = f"p_{i}[0] * (1 - t)^{n-i} * t^{i}"
        term_y = f"p_{i}[1] * (1 - t)^{n-i} * t^{i}"
        term_z = f"p_{i}[2] * (1 - t)^{n-i} * t^{i}"
        
        # 合并到最终方程
        equations.append((term_x, term_y, term_z))
    
    return equations