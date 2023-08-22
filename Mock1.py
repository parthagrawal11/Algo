arr = [10,20,30,40]
# a = [1,2,3,4]
# a = [-1,-2,-3,-4]
# a = [-1 ,0,-3,-4]
# a = [0,0,-3,-4]
#out--> [24,12,8,6]
def my_creation(a):
    out=[]
    for n,i in enumerate(a):
        p=1
        for m,j in enumerate(a):
            if m==n:
                continue
            print(n,i,j)
            p=j*p
        out.append(p)
    return out

def get_product_array(arr):
    n = len(arr)
    left, right = [1]*n, [1]*n
    product_array=[]
    for i in range(1,n):
        print(i)
        left[i]=left[i-1]*arr[i-1]
    for i in range(1,n):
        right[i]=right[i-1]*arr[::-1][i-1]
    for i in range(n):
        product_array.append(left[i]*right[::-1][i])
    return product_array    

out = get_product_array(arr)
# out = my_creation(a)
print(out)