import numpy as np
import pandas as pd


class Tree:
    def __init__(self):
        self.attribute = None
        self.children = None
        self.value = None  # add value component to build tree method
        self.classification = None

    def set_classification(self, classification):
        self.classification = classification

    def set_attribute(self, attribute):
        self.attribute = attribute
        self.children = {}

    def add_child(self, key, child):
        self.children[key] = child

    def get_child(self, key):
        if self.children is not None:
            return self.children.get(key)

    def has_children(self):
        if self.children is None:
            return False
        return True

    def get_classification(self):
        return self.classification

    def get_attribute(self):
        return self.attribute

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value


pd.set_option("display.max_rows", None, "display.max_columns", None)


def loadData(year):
    # load in passing data
    df1 = pd.read_csv("Raw Data/Offense passing/" + year + "nflstats.csv")
    df1.drop(['Team', 'TD', 'Lng'], axis=1, inplace=True)
    df1.rename(
        columns={'Att': 'Pass Att', '1st': 'Pass 1st', '1st%': 'Pass 1st%', '20+': '20+ (Pass)', '40+': '40+ (Pass)'},
        inplace=True)

    # load in rushing data
    df2 = pd.read_csv("Raw Data/Offense Rushing/" + year + "rushing.csv")
    df2.drop(['Team', 'TD', 'Lng', 'Rush FUM'], axis=1, inplace=True)
    df2.rename(columns={'Att': 'Rush Att', '20+': '20+ (Rush)', '40+': '40+ (Rush)'}, inplace=True)

    # load in scoring data
    df3 = pd.read_csv("Raw Data/Scoring/" + year + "scoring.csv")
    df3.drop(['Team', '2-PT'], axis=1, inplace=True)

    # return
    df = pd.concat([df3, df2, df1], axis=1)
    return df


df = pd.concat([loadData("2010"), loadData("2011"), loadData("2012"), loadData("2013"), loadData("2014"),
                loadData("2015"), loadData("2016"), loadData("2017"), loadData("2018"), loadData("2019")]).reset_index(
    drop=True)




# checks the purity of the data
def isPure(data):
    playoffClasses = data[data.columns[-1]]
    unique = np.unique(playoffClasses)
    if len(unique) != 1:
        return False
    else:
        return True


# classifies the data
def classifier(data):
    playoffClasses = data[data.columns[-1]]
    unique_statuses, unique_statuses_counts = np.unique(playoffClasses, return_counts=True)

    index = unique_statuses_counts.argmax()

    classification = unique_statuses[index]

    return classification


# finds potential splits
def potential_splits(data, used):
    Psplits_holder = {}
    _, numColumns = data.shape
    for cIndex in range(1, numColumns - 1):
        if data.columns[cIndex] not in used:
            Psplits_holder[cIndex] = []
            vals = data.iloc[:, cIndex]
            uniqueVals = np.unique(vals)

            for i in range(len(uniqueVals)):
                if i != 0:
                    curVal = uniqueVals[i]
                    prevVal = uniqueVals[i - 1]
                    pSplit = (curVal + prevVal) / 2

                    Psplits_holder[cIndex].append(pSplit)
    return Psplits_holder


# splits the data based on the splitColVal
def splitter(data, split_column, split_val):
    splitColVal = data.iloc[:, split_column]

    below = data[splitColVal <= split_val]
    above = data[splitColVal > split_val]
    return below, above


def calcEntropy(data):
    labelCol = data.iloc[:, -1]
    _, nums = np.unique(labelCol, return_counts=True)
    probabilitiesArr = nums / nums.sum()
    entropy = sum(probabilitiesArr * -np.log2(probabilitiesArr))

    return entropy


def calcOverallEntropy(below, above):
    numDPoints = len(below) + len(above)
    probBelow = len(below) / numDPoints
    probAbove = len(above) / numDPoints
    # overall
    overall = ((probBelow * calcEntropy(below)) + (probAbove * calcEntropy(above)))

    return overall


def bestSplitChooser(data, used):
    splits = potential_splits(data, used)
    overallEntropy = 999
    for cIndex in splits:
        for i in splits[cIndex]:
            below, above = splitter(data, split_column=cIndex, split_val=i)
            curOverallEntropy = calcOverallEntropy(below, above)

            if curOverallEntropy <= overallEntropy:
                overallEntropy = curOverallEntropy
                bestColSplit = cIndex
                bestValSplit = i

    return bestColSplit, bestValSplit


# builds tree from a dataframe
def build_tree(data, used):
    data.reset_index(drop=True, inplace=True)
    tree = Tree()
    if isPure(data):
        tree.set_classification(data.iloc[:, -1][0])
    elif len(used) != data.shape[1] - 1:
        col, val = bestSplitChooser(data, used)
        attribute = data.columns[col]
        tree.set_attribute(attribute)
        tree.set_value(val)
        below, above = splitter(data, col, val)
        used.append(attribute)
        if len(above) != 0:
            tree.add_child("above", build_tree(above, used))
        if len(below) != 0:
            tree.add_child("below", build_tree(below, used))
        used.pop()
    else:
        tree.set_classification(data.mode(data.iloc[:, -1][0]))
    return tree


# construct 50 trees based on random subsets of the original data frame -- use 8 random columns for each


tree = build_tree(df.drop(df.index[288:320]), [])


def test(tree, data_row):
    current_tree = tree
    while current_tree.get_classification() is None:
        attribute = current_tree.get_attribute()
        if data_row[attribute] <= current_tree.get_value():
            current_tree = current_tree.get_child("below")
        else:
            current_tree = current_tree.get_child("above")

    return current_tree.get_classification()


print(test(tree, df.drop(['PlayoffStatus'], axis=1).iloc[0]))

for i in range(288, 320):
    print("Team "+str(i-288)+": "+test(tree, df.drop(['PlayoffStatus'], axis=1).iloc[i]))


