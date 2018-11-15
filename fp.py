//对前七天的商品的规则进行挖掘
//(id1,id2,id3,....,idk)---一次购买行为的所有商品id

                                                                                                                                                                                             
       

具体pyspark 代码如下：



from pyspark.mllib.fpm import FPGrowth
from pyspark.sql import SparkSession
spark = SparkSession \
.builder \
.master("local")\
.appName("RDD_and_DataFrame") \
.config("spark.some.config.option", "some-value") \
.getOrCreate()

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
#执行sql语句，将订单数据整理成一个个事务
sentenceData = spark.sql("读取数据")
#将Dataframe类型转换成rdd进行map
rdd=sentenceData.rdd
#将数据格式转换成[[itemid,itemid,...],[itemid,itemid,...],[itemid,itemid,...]]，[itemid,itemid,...]表示一组事务，也就是同以订单中的物品组合
rdd=rdd.map(lambda x: x[1].split(",")).collect()
#转换成模型需要的数据格式rdd类型，新版本可能模型有变化
rdd=spark.sparkContext.parallelize(rdd)
#后面的参数分别表示最低支持度，2 表示输入的rdd 分成几个partition来做处理
model = FPGrowth.train(rdd, 0.00001, 2)
skuresult=model.freqItemsets().collect()
#skuresult为从订单中挖掘出来的频繁项集



2.关联规挖掘（频繁项集）
根据频繁项集找到对应的强关联规则：
找到k项频繁项集，将k项频繁项集进行关联规则提取，1对1,1对多，多对多
k项频繁项集假设为{1,2,3}
具体步骤：
step1:求出所有非空子集{1}，{2}，{3}，{1,2}，{2,3}，{1,3}
step2:任意取两个不想交的非空子集生成一个规则：如{1}--》{2,3}
step3:判断规则是否为强关联规则，如果是则进行存储规则表中可用dict存储以便于后续用来进行匹配推荐。
step4:循环2,3步


具体python代码如下：

from pyspark.sql import SparkSession
from pyspark.mllib.fpm import FPGrowth
from pyspark.sql import SparkSession
spark = SparkSession \
.builder \
.master("local")\
.appName("RDD_and_DataFrame") \
.config("spark.some.config.option", "some-value") \
.getOrCreate()

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
sentenceData = spark.sqls(‘’‘)
data=sentenceData.rdd.map(lambda x: x[1].split(",")).collect()
rdd=spark.sparkContext.parallelize(data)
model = FPGrowth.train(rdd, 0.00001, 2)
result=model.freqItemsets().collect()
ordernum=len(data)+1
ss=calrule(result)

def PowerSetsBinary(items):
N = len(items)
set_all=[]
for i in range(2**N):
combo = []
for j in range(N):
if(i >> j ) % 2 == 1:
combo.append(items[j])
if len(combo)>0 and len(combo)<len(items) :
set_all.append(combo)
return set_all


def rules(items):
out = PowerSetsBinary(items)
listrules=[]
for i in range(len(out)):
lista=out[i]
for j in range(len(out)):
listb = out[j]
if j==i or set(lista)&set(listb):
continue
else:
listtemp = []
listtemp.append(lista)
listtemp.append(listb)
listrules.append(listtemp)
return listrules


def getfreq (result,lista):
for i in result:
if set(i.items) ==set(lista):
return i.freq
return 0

def getsupprot(result, lista, ordernum):
afreq = getfreq(result, lista)
support = float(afreq / float(ordernum))
return support


def getConfidence(result, ordernum, lista, listb):
listab = list(set(lista + listb))
Sab = getsupprot(result, listab, ordernum)
Sa = getsupprot(result, lista, ordernum)
confidence = Sab / float(Sa)
return confidence


def strongeRule2(result, ordernum, lista, listb, minConfidence=0.001, minsupport=0.02):
confidenceab = getConfidence(result, ordernum, lista, listb)
listab = list(set(lista + listb))
supportab = getsupprot(result, listab, ordernum)
if supportab >= minsupport and confidenceab >= minConfidence:
return [lista, listb,confidenceab,supportab ]

else :
return 0
#计算出提升度
def getlift(result, ordernum, lista, listb):
listab = list(set(lista + listb))
# print(listab)
Sab = getsupprot(result, listab, ordernum)
Sa = getsupprot(result, lista, ordernum)
Sb = getsupprot(result, listb, ordernum)
liftab=0
if Sb!=Sab:
liftab = float(((1 - Sa) * Sab) )/ float((Sa * (Sb - Sab)))
return liftab


def strongeRule(result, ordernum, lista, listb, minConfidence=0.001, minsupport=0.02,minlift=1):
listab = list(set(lista + listb))
supportab = getsupprot(result, listab, ordernum)
confidenceab = getConfidence(result, ordernum, lista, listb)
liftab = getlift(result, ordernum, lista, listb)
if supportab >= minsupport and confidenceab >= minConfidence and liftab >= minlift:
return [lista, listb,confidenceab,supportab,liftab ]
else:
return 0


def calrule(result):
l=[]
for i in range(len(result)):
if len(result[i][0])>1:
listout= rules(result[i][0])
for i in listout:
flag=0
for j in l:
if (set(i[0])==set(j[0])) and (set(i[1])==set(j[1])):
flag=1
break
if flag==0:
t = strongeRule2(result, ordernum, i[0], i[1], 0.00006, 0.0002)
if t:
l.append(t)
return l




3、应用于推荐：
根据用户的购物车数据，利用强关联规则进行补充，扩大推荐商品。
若购物车中没有商品，可以根据前一到三次的购物记录作为购物车数据，进行强关联规则的推荐。
假设要推k件商品，（兜底：根据商品的销售频次推荐topk---非常简单的推荐）
（1）根据挖掘出来的频繁相机进行推荐，
 针对新用户（购物车无信息）采用兜底
 针对老用户（购物车有商品信息）：
        a.直接推购物车商品中的前k件热门商品
        b.根据购物车商品进行补充列表，将购物车中的商品相关的频繁项集进行挖掘，找到相关的商品进行推荐


