# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def verticalnode(root,x,index):
    if root.left:
        new_x= x-1
        if index[new_x]:
            index[new_x].append(root.left)
        else:
            index[new_x]=[root.left]
        verticalnode(root,new_x,index)
    if root.right:
        new_x= x-1
        if index[new_x]:
            index[new_x].append(root.right)
        else:
            index[new_x]=[root.right]
        verticalnode(root,new_x,index)
    return index

class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        res =[]
        index = Index
        # index ={}
        if root:
            print(root.left)
            Index
            res = self.verticalTraversal(root.left)
            res.append(root.val)
            res = res+self.verticalTraversal(root.right)
        return res

root =  TreeNode{val: 3, left: TreeNode{val: 9, left: None, right: None}, right: TreeNode{val: 20, left: TreeNode{val: 15}, right: TreeNode{val: 7}}

x=0
index = {}

index = verticalnode(root,x,index)
