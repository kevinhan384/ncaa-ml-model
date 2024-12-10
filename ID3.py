from node import Node
from collections import defaultdict
import math

#helper function to calculate entropy
def _entropy(fields):
  # Globals
  d = defaultdict(int)
  n = len(fields)
  
  for field in fields:
    d[field["Placing"]] += 1
  
  return sum([-(count/n) * math.log2(count/n) for count in d.values()])

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  # Step 1
  t = Node()
  
  # Step 2
  d = {}
  for example in examples:
    d[example["Placing"]] = d.get(example["Placing"], 0) + 1
  keyVals = list(d.items())
  keyVals.sort(reverse=True, key=lambda x: x[1])
  t.label = keyVals[0][0]

  # Step 3
  if len(keyVals) == 1:
    return t
  
  # Step 4
  # check any attributes left to split dataset by
  # examples always has class attribute, which is why - 1 
  if len(examples[0]) - 1 == 0:
    t.label = default
    return t
  
  # Step 5 
  # Calculate information gain for each attribute, choose attribute yielding highest information gain
  info_gain = -1
  decision_attribute = None
  parent_entropy = _entropy(examples)
  
  #all possible values for every attribute
  possible_vals = defaultdict(set)
  for example in examples:
    for key, val in example.items():
      if key == "Placing" or key == "Rk" or key == "School" or  val == "?":
        continue
      possible_vals[key].add(val)
  
  #calculate post-split entropy using conditional entropy ????
  for attribute in possible_vals.keys():
        
    #split by this attribute, append row 
    splits = defaultdict(list)
    missing_vals = 0
    for example in examples:
      if example[attribute] == "?":
        missing_vals += 1
      else:
        value = example[attribute]
        splits[value].append(example)
    
    #don't count fields with empty attribute 
    total_instances = len(examples) - missing_vals
    child_entropy = 0
    
    for split in splits.values():
      weight = len(split) / total_instances
      child_entropy += weight * _entropy(split)
    
    if parent_entropy - child_entropy > info_gain:
      info_gain = parent_entropy - child_entropy
      decision_attribute = attribute
  
  #set attribute to split by for current node
  #add to used attributes to avoid infinite loop
  t.decision_attribute = decision_attribute
  # Step 6 recursive call and add child nodes
  splits = defaultdict(list)
  for example in examples:
    value = example[decision_attribute]
    if value != "?":
      example_copy = example.copy()
      example_copy.pop(decision_attribute)
      splits[value].append(example_copy)
  
  #split into child nodes 
  for value,split in splits.items():
    if split != None:
      t.children[value] = ID3(split, t.label)
  return t

def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''
  #leaf node, just return 
  if not node.children:
    return node
  
  original_accuracy = test(node, examples)
  
  def helper(pruned, examples):
    #base case
    if not pruned.children:
      return pruned
    
    #recurse for each child
    for val in pruned.children:
      pruned.children[val] = helper(pruned.children[val], examples)
      
    #create copy of children in case need to restore
    children_copy = pruned.children.copy()
    pruned.children = {}
    curr_accuracy = test(node, examples)
    
    #determine if should remove children or not based on accuracy improvement
    if curr_accuracy > original_accuracy:
      return pruned
    else:
      pruned.children = children_copy
      return pruned
  
  return helper(node, examples)


#simply compare predicted vs actual and calculate percentage we get right
def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  correct = 0

  for example in examples:
    predicted = evaluate(node, example)  # get prediction
    actual = example["Placing"]  # get actual

    if predicted == actual:
      correct += 1

  #get success rate
  accuracy = correct / len(examples)
  return accuracy


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  #traverse tree until no more children, i.e. leaf
  root = node
  while root.children:
    attribute = root.decision_attribute
    example_attribute_val = example[attribute]
    
    #attribute may be an unseen value, in which case just return current node.label
    if example_attribute_val in root.children:
      root = root.children[example_attribute_val]
    else:
      return root.label

  return root.label