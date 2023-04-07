import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from matplotlib import pyplot as plt
# matplotlib设置字体为黑体，不显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# pandas读取数据建立matplotlib可视化图表
assignment6_3 = pd.read_csv('./6_3.csv', encoding='gbk')
pplot = sm.ProbPlot(assignment6_3['重量'], fit=True)
plt.subplots(1, 2, figsize=(12, 5))

# 作Q-Q图
ax1 = plt.subplot(121)
pplot.qqplot(line='r', ax=ax1, xlabel='期望正态值', ylabel='标准化的观测值')
ax1.set_title('正态Q-Q图', fontsize=15)

# 作P-P图
ax2 = plt.subplot(122)
pplot.ppplot(line='45', ax=ax2, xlabel='期望的累积概率', ylabel='观测的累计概率')
ax2.set_title('正态P-P图', fontsize=15)

# 显示图表
plt.show()

# 检验
W, pw_value = stats.shapiro(assignment6_3['重量'])
print(f"统计量W = {W: .5f}, p值 ={pw_value: .4g}")
d = assignment6_3['重量']
D, pd_value = stats.kstest(d, 'norm', alternative='two-sided', mode='asymp', args=(d.mean(), d.std()))
print(f"统计量D = {D: .5f}, p值 ={pd_value: .4g}")
# 自己添加的结论
print("由于p值都大于0.05，不拒绝原假设，没有证据表明金属板的重量不服从正态分布")
