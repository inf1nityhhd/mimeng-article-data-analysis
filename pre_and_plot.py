import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from snownlp import SnowNLP
import seaborn as sns
import jieba
import re
import os

sns.set(style='darkgrid')
# 设置pandas
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# ----------------------------------- 数据探索与预处理 ---------------------------------------------
# 数据读取
data = pd.read_excel('./datas/咪蒙阅读数据.xlsx', index_col=0)
# 探索数据
# print(data.info())
# print(data.head())
# print(data.describe())
# print(data.keys())
data.drop(['位置', '链接', '原文链接', '赞赏', '编号', '公众号'],
          axis=1,
          inplace=True
          )
# data = data.loc[:, ['标题', '发文时间', '点赞', '阅读']] 有用的列
# 删除标题为空的行
data.dropna(subset=['标题'], inplace=True)
# print(data['标题'].value_counts())
# 填充'作者'列的空值
data['作者'].fillna('咪蒙', inplace=True)
# # 处理后输出结果
# print(data.info())
# # 保存数据
# data.to_excel('./datas/after_pre.xlsx')
# data.to_csv('./datas/after_pre.csv', index=False)
# ----------------------------------- 文章原创情况 ---------------------------------------------
# sns.set(style='darkgrid')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# self_made_count = 0
# others_count = 0
# for title in data['作者']:
#     if '咪蒙' in str(title):
#         self_made_count = self_made_count + 1
#     else:
#         others_count = others_count + 1
# plt.title('文章原创情况')
# plt.pie([self_made_count, others_count],
#         labels=['原创', '非原创'],
#         explode=(0.2, 0.0),
#         autopct='%1.0f%%',
#         )
# plt.legend(['原创', '非原创'])
# plt.savefig('./images/文章原创情况.jpg', bbox_inches='tight')
# plt.show()
# print(self_made_count, others_count)
# ----------------------------------- 10万+文章比例 ---------------------------------------------
# upper_10 = (data['阅读'] < 100000).value_counts()[0]
# lower_10 = (data['阅读'] < 100000).value_counts()[1]
# print('10万+文章数量：', upper_10)
# print('非10万+文章数量：', lower_10)
# ----------------------------------- 15大爆款文章 ---------------------------------------------
# sns.set(style='whitegrid')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# top_article = pd.DataFrame(data)
# top_article.sort_values(by='点赞', ascending=False, inplace=True, axis=0)
# plt.figure(figsize=(10, 9))
# sns.barplot(x='点赞', y='标题', data=top_article.head(15), orient='h')
# plt.title('TOP15文章', fontsize=18)
# plt.xlabel('点赞数', fontsize=15)
# plt.ylabel('文章标题', fontsize=15)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=16)
# plt.savefig('./images/15大爆款文章.jpg', bbox_inches='tight')
# plt.show()
# ----------------------------------- 10万+爆款文章统计 ---------------------------------------------
# plt.rcParams['font.sans-serif'] = ['SimHei']
# sns.set(style='darkgrid')
# data_upper_tenmillion = pd.DataFrame(data)
# -----------------------------------统计常用词汇并绘制词云----------------------------------------------
# # print(data.head())
# titles = data['标题'].where(data['标题'] != '分享图片').dropna()
# # 将标题分词处理
# titles_splited = []
# # 正则表达式去除标点符号
# reg = u"[\s+\.\!\/_,$%^*(+\"\'0-9]+|[+——！，。？、~@#￥%……&*（）]+"
# for title in titles:
#     title = re.sub(reg, '', title)
#     # 分词
#     title = jieba.lcut(title, cut_all=True)
#     titles_splited.append(title)
# # 初始化停用词列表
# stopwords = []
# with open('./utils/stopwords.txt', 'r', encoding='utf-8') as f:
#     for word in f.readlines():
#         word = word.strip()
#         stopwords.append(word)
# # 统计标题中的常用词汇
# title_word_count = {}
# for word_list in titles_splited:
#     for word in word_list:
#         if word in stopwords:
#             break
#         if word not in title_word_count:
#             title_word_count[str(word)] = 1
#         else:
#             title_word_count[str(word)] = title_word_count[str(word)] + 1
#
# # 统计前50个关键词并绘制词云
# top_words = dict(sorted(title_word_count.items(),
#                         key=lambda v: v[1],
#                         reverse=True)[0:50]
#                 )
# mask = plt.imread('./utils/mask.jpg')
# wordcloud = WordCloud(font_path="./fonts/幼圆.TTF",
#                       mask=mask, scale=8,
#                       background_color='white').generate_from_frequencies(top_words)
# WordCloud.to_file(wordcloud, './images/标题关键字词云.jpg')
# -----------------------------------标题情感分析并绘制比例饼图----------------------------------------------
# titles = data['标题']
# emotions = {'positive': 0, 'neutral': 0, 'negative': 0}
# for title in titles:
#     score = SnowNLP(str(title)).sentiments
#     if score > 0.6:
#         emotions['positive'] = emotions['positive'] + 1
#     elif score < 0.5:
#         emotions['negative'] = emotions['negative'] + 1
#     else:
#         emotions['neutral'] = emotions['neutral'] + 1
# # print(emotions)
# plt.figure(figsize=(5, 4))
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.title('文章标题的情感极性分布', fontsize=14)
# plt.pie(x=emotions.values(), labels=['正面', '中性', '负面'],
#         explode=(0.08, 0.05, 0.07),
#         autopct='%1.2f%%',
#         startangle=np.random.randint(0, 360))
# plt.legend(['正面', '中性', '负面'])
# plt.show()
# plt.savefig('./images/文章标题的情感极性分布.jpg')
# -----------------------------------按季度统计文章发表的数量及点赞数，绘制相关图像---------------------------------------------
# sns.set(style='darkgrid')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# season_data = pd.DataFrame(data)
# season_data.set_index(season_data['发文时间'], inplace=True)
# plt.figure(figsize=(9, 6))
# labels = ['2015年第三季度', '2015年第四季度',
#           '2016年第一季度', '2016年第二季度', '2016年第三季度', '2016年第四季度',
#           '2017年第一季度', '2017年第二季度', '2017年第三季度', '2017年第四季度',
#           '2018年第一季度', '2018年第二季度', '2018年第三季度', '2018年第四季度',
#           '2019年第一季度'
#           ]
# quarter_data_like = pd.DataFrame(season_data.resample('Q')['点赞'].sum())
# plt.ylabel('点赞数量')
# plt.xlabel('2015年第三季度到2019年第一季度')
# plt.xticks(ticks=quarter_data_like.index, labels=labels, rotation=45)
# plt.title('每年各季度点赞总数的变化趋势图')
# # sns.distplot(quarter_data_like['点赞'])
# plt.bar(quarter_data_like.index, quarter_data_like['点赞'], width=1, color='cornflowerblue')
# plt.plot(quarter_data_like, marker='.', c='orange')
# plt.legend(['点赞'])
# plt.savefig('./images/每年各季度点赞总数的变化趋势图.jpg', bbox_inches='tight')
# plt.show()
# plt.figure(figsize=(9, 6))
# labels = ['2015年第三季度', '2015年第四季度',
#           '2016年第一季度', '2016年第二季度', '2016年第三季度', '2016年第四季度',
#           '2017年第一季度', '2017年第二季度', '2017年第三季度', '2017年第四季度',
#           '2018年第一季度', '2018年第二季度', '2018年第三季度', '2018年第四季度',
#           '2019年第一季度'
#           ]
# quarter_data_count = pd.DataFrame(season_data.resample('Q')['点赞'].count())
# plt.ylabel('文章数量')
# plt.xlabel('2015年第三季度到2019年第一季度')
# plt.xticks(ticks=quarter_data_count.index, labels=labels, rotation=45)
# plt.title('每年各季度文章发表总数的变化趋势图')
# plt.bar(quarter_data_count.index, quarter_data_count['点赞'], width=1, color='cornflowerblue')
# plt.plot(quarter_data_count, marker='.', c='orange')
# plt.legend(['文章'])
# plt.savefig('./images/每年各季度文章发表总数的变化趋势图.jpg', bbox_inches='tight')
# plt.show()
# -----------------------------------统计其文章发表的主要时间段---------------------------------------------
# sns.set(style='darkgrid')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.figure(figsize=(7, 5))
# hour_data = pd.DataFrame(data)
# hour_data.set_index(hour_data['发文时间'], inplace=True)
# hour_data_count = pd.DataFrame(hour_data.groupby(hour_data.index.hour)['标题'].count())
# hour_data_count.rename({'标题': '标题数量'}, axis=1, inplace=True)
# plt.title('发文时间-文章数量关系', fontsize=15)
# plt.xlabel('发文时间', fontsize=13)
# plt.ylabel('文章数量', fontsize=13)
# labels = [str(hour) + '时' for hour in list(hour_data_count.index)]
# plt.xticks(ticks=hour_data_count.index, labels=labels)
# plt.bar(hour_data_count.index, height=hour_data_count['标题数量'])
# for x, y in zip(hour_data_count.index, hour_data_count['标题数量']):
#     plt.text(x, y + 0.05, '%d' % y, ha='center', va='bottom', fontsize=10)
# plt.savefig('./images/发文时间与文章数量关系图.jpg')
# plt.show()
# ----------------------------------- 点赞数随时间变化的情况 ---------------------------------------------
# sns.set(style='darkgrid')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# day_data = pd.DataFrame(data)
# plt.figure(figsize=(9,7))
# sns.lineplot(x='发文时间', y='点赞', data=day_data)
# plt.xticks(rotation=45)
# plt.title('点赞数随发文时间的变化')
# plt.ylabel('点赞数量')
# plt.xlabel('发文时间')
# plt.legend(['点赞'])
# plt.savefig('./images/点赞数随时间变化的情况.jpg', bbox_inches='tight')
# plt.show()
# ----------------------------------- 不同点赞数区间内的文章分布情况 ---------------------------------------------
# plt.rcParams['font.sans-serif'] = ['SimHei']
# data_article_like_count = pd.DataFrame(data)
# # 统计不同点赞区间内的文章数量分布情况
# bin = [0, 5000, 20000, 40000, 100001]
# data_article_like_count = pd.cut(data_article_like_count['点赞'], bins=bin, right=False).value_counts()
# # plt.figure(figsize=(11, 7))
# plt.title('文章点赞区间分布', fontsize=15)
# plt.xlabel('点赞数', fontsize=13)
# plt.ylabel('文章总数', fontsize=13)
# data_article_like_count.plot(kind='bar', figsize=(6, 5))
# plt.savefig('./images/不同点赞数区间内的文章分布情况.jpg', bbox_inches='tight')
# plt.show()
# ----------------------------------- 文章字数分布 ---------------------------------------------
# sns.set(style='darkgrid')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# word_count = {'0~1000': 0, '1000~2000': 0, '2000~3000': 0, '3000~4000': 0, '4000~': 0}
# articles = data['标题']
# for article in os.listdir('./datas/articles/'):
#     print(article)
#     f = open('./datas/articles/' + article, 'r', encoding='utf-8')
#     article_len = int(len(''.join(f.readlines())))
#     f.close()
#     if article_len < 1000:
#         word_count['0~1000'] = word_count['0~1000'] + 1
#     elif article_len < 2000:
#         word_count['1000~2000'] = word_count['1000~2000'] + 1
#     elif article_len < 3000:
#         word_count['2000~3000'] = word_count['2000~3000'] + 1
#     elif article_len < 4000:
#         word_count['3000~4000'] = word_count['3000~4000'] + 1
#     else:
#         word_count['4000~'] = word_count['4000~'] + 1
# words_count = pd.DataFrame(
#     data=word_count.values(),
#     index=word_count.keys(),
#     columns=['字数']
# )
# sns.barplot(x=words_count.index, y='字数', data=words_count)
# plt.xlabel('字数')
# plt.ylabel('文章数量')
# plt.title('文章字数分布')
# plt.show()
# plt.savefig('./images/文章字数分布.jpg', bbox_inches='tight')
