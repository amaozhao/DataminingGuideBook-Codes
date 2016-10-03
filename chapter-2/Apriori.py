# coding: utf-8
import os
import pandas as pd
from collections import defaultdict
from operator import itemgetter

ratings_filename = os.path.join('BX-Dump', "BX-Book-Ratings.csv")
all_ratings = pd.read_csv(
    ratings_filename,
    sep=";",
    header=None,
    names=["userId", "bookId", "rating"]
)
all_ratings["favorable"] = all_ratings["rating"] > 5
# 从数据集中选取一部分数据用作训练集,这能有效减少搜索空间. 取前 200 名用户的打分数据
ratings = all_ratings[all_ratings['userId'].isin(range(10000))]
favorable_ratings = ratings[ratings["favorable"]]  # 新建一个数据集,只包括用户喜欢某部电影的数据行
favorable_reviews_by_users = dict(
    (
        k, frozenset(v.values)
    ) for k, v in favorable_ratings.groupby("userId")["bookId"])

# 在生成项集时, 需要搜索用户喜欢的电影。
# 因此, 需要知道每个用户各喜欢哪些电影,
# 按照userId进行分组, 并遍历每个用户看过的每一部电影(frozenset比列表的操作更快)
num_favorable_by_movie = ratings[["bookId", "favorable"]].groupby(
    "bookId").sum()  # 创建一个数据框,以便了解每部电影的影迷数量
num_favorable_by_movie.sort_values("favorable", ascending=False)[
    :5]  # 查看最受欢迎的五部电影


def find_frequent_itemsets(
        favorable_reviews_by_users, k_1_itemsets, min_support):
    counts = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):
                for other_reviewed_movie in reviews - itemset:
                    current_superset = itemset | frozenset(
                        (other_reviewed_movie,))
                    counts[current_superset] += 1
    return dict(
        [
            (itemset, frequency) for itemset, frequency in counts.items(
            ) if frequency >= min_support
        ]
    )


frequent_itemsets = {}  # 频繁项集字典, 便于长度查找
min_support = 4

frequent_itemsets[1] = dict(
    (
        frozenset(
            (bookId,)
        ),
        row["favorable"]
    ) for bookId, row in num_favorable_by_movie.iterrows(
    ) if row["favorable"] > min_support
)

print(u"共有{}书籍有超过{}喜欢标注".format(len(frequent_itemsets[1]), min_support))

for k in range(2, 20):
    # 创建循环, 运行Apriori算法,存储算法运行过程中发现的新项集
    # 循环体中, k表示即将发现的频繁项集的长度,
    # 用键k-1可以从 frequent_itemsets字典中获取刚发现的频繁项集.
    # 新发现的频繁项集以长度为键,将其保存到字典中
    cur_frequent_itemsets = find_frequent_itemsets(
        favorable_reviews_by_users,
        frequent_itemsets[k - 1],
        min_support
    )
    if len(cur_frequent_itemsets) == 0:
        print(u"未找到长度为{}的频繁项集".format(k))
        break
    else:
        print(u"共找到{}条频繁项集对应于长度{}".format(
            len(cur_frequent_itemsets), k))
        frequent_itemsets[k] = cur_frequent_itemsets
# 删除长度为 1 的项集(对只有一个元素的项集不再感兴趣)
del frequent_itemsets[1]

# 从频繁项集中抽取出关联规则,把其中几部电影作为前提,
# 另一部电影作为结论组成如下形式的规则:如果用户喜欢前提中的所有电影,那么他们也会喜欢结论中的电影
candidate_rules = []
for itemset_length, itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():
        for conclusion in itemset:
            premise = itemset - set((conclusion,))
            candidate_rules.append((premise, conclusion))
print(u"共有{}候选规则".format(len(candidate_rules)))

# 计算每条规则的置信度
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in favorable_reviews_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1
rule_confidence = {
    candidate_rule: correct_counts[
        candidate_rule] / float(
            correct_counts[
                candidate_rule
            ] + incorrect_counts[
                candidate_rule]
    ) for candidate_rule in candidate_rules
}

sorted_confidence = sorted(
    rule_confidence.items(),
    key=itemgetter(1),
    reverse=True
)

# 显示电影名称
book_name_filename = os.path.join('BX-Dump', "BX-Books.csv")
book_name_data = pd.read_csv(
    book_name_filename,
    sep=";",
    header=None
)
book_name_data.columns = [
    "bookId", "Title", "Author", "Publish Date",
    "Press", "img1", "img2", "img3"
]


def get_book_name(book_id):
    title_object = book_name_data[
        book_name_data["bookId"] == book_id]["Title"]
    title = title_object.values[0]
    return title


for index in range(5):
    print(u"规则 #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ", ".join(get_book_name(idx) for idx in premise)
    conclusion_name = get_book_name(conclusion)
    print(
        u"-规则: 如果某人喜欢{0}他们也会喜欢{1}"
        .format(premise_names, conclusion_name))
    print(
        u"-置信度: {0:.3f}".format(
            rule_confidence[(premise, conclusion)]))
    print("")


# 测试
test_dataset = all_ratings[~all_ratings['userId'].isin(range(200))]
test_favorable = test_dataset[test_dataset["favorable"]]
test_favorable_by_users = dict(
    (
        k,
        frozenset(v.values)
    ) for k, v in test_favorable.groupby("userId")["bookId"]
)
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in test_favorable_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1

test_confidence = {
    candidate_rule: correct_counts[
        candidate_rule] / float(
        correct_counts[
            candidate_rule] + incorrect_counts[
            candidate_rule]) for candidate_rule in rule_confidence}
sorted_test_confidence = sorted(
    test_confidence.items(),
    key=itemgetter(1),
    reverse=True
)
print(sorted_test_confidence[:5])

for index in range(10):
    print(u"规则 #{0}".format(index + 1))
    premise, conclusion = sorted_confidence[index][0]
    premise_names = ", ".join(get_book_name(idx) for idx in premise)
    conclusion_name = get_book_name(conclusion)
    print(u"规则: 如果某人喜欢{0}, 他们也会喜欢{1}".format(
        premise_names, conclusion_name))
    print(u"- 训练集: {0:.3f}".format(
        rule_confidence.get((premise, conclusion), -1)))
    print(u"- 测试集: {0:.3f}".format(
        test_confidence.get((premise, conclusion), -1)))
    print("")
