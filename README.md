### EDA
Heatmap
![heatmap.png](graphs%2Fheatmap.png)

Boruta
![boruta of all data.png](graphs%2Fboruta%20of%20all%20data.png)

### 第一题
performance（momentum）和是否破发非常相关
![point vs performance vs server.png](graphs%2Fpoint%20vs%20performance%20vs%20server.png)

momentum基本反应得分情况
![point vs sigma_point vs sigma_performance.png](graphs%2Fpoint%20vs%20sigma_point%20vs%20sigma_performance.png)


#
### 第二题

不一定有用的图，可以证明胜率和得分情况的积分很相关（注意胜率的尺度被扩大了10倍）
![point vs sigma_win_probability.png](graphs%2Fpoint%20vs%20sigma_win_probability.png)

momentum->破发球->胜率改变->swing(momentum->破发球 同第一题散点图)

破发球->胜率变化->swing
![win probability vs partial win probability vs break serve.png](graphs%2Fwin%20probability%20vs%20partial%20win%20probability%20vs%20break%20serve.png)

boruta衡量performance和momentum与其他变量相关性（推迟）

#
### 第三题
boruta/pca/attention权重

#
## 模型

### GLAC微观模型
宏观-微观图
![micro vs macro.png](graphs%2Fmicro%20vs%20macro.png)

预测的momentum和win rate求导的图
![predicted momentum vs partial win probaility.png](graphs%2Fpredicted%20momentum%20vs%20partial%20win%20probaility.png)

预测和实际的momentum对比图
![predicted momentum vs momentum.png](graphs%2Fpredicted%20momentum%20vs%20momentum.png)

#
### in-match set win rate prediction model局内预测模型

预测的momentum和局内预测模型对比图

女子比赛数据？红土比赛数据？

#
### elo model局外预测模型

#
### 宏观模型

使用某player数据（多个match数据）画出实际momentum，预测momentum，并提供建议

