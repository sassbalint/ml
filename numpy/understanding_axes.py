import numpy as np

print("""
I managed to understand how numpy axes work from this wording:
"axis 0 is the outermost axis indexing the largest subarrays
 while axis n-1 is the innermost axis indexing individual elements."
Or put it another way:
x.sum(axis=0) is x[sum][*][*] ~ just do,
x.sum(axis=1) is x[*][sum][*] ~ take all and do,
x.sum(axis=2) is x[*][*][sum] ~ take all and take all and do.
cf. https://stackoverflow.com/questions/24281263
""")

def out(obj, msg):
    print(f'{msg} =\n{obj}\n')

x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

out(x, 'x')

out(np.sum(x, axis=0), 'sum/0')
out(np.sum(x, axis=1), 'sum/1')
out(np.sum(x, axis=2), 'sum/2')

print('m(<n) means which was originally axis=n is now axis=m\n')
out(np.sum(np.sum(x, axis=0), axis=0), 'sum/0,0(<1)')
out(np.sum(np.sum(x, axis=1), axis=0), 'sum/1,0')
out(np.sum(np.sum(x, axis=0), axis=1), 'sum/0,1(<2)')
out(np.sum(np.sum(x, axis=2), axis=0), 'sum/2,0')
out(np.sum(np.sum(x, axis=1), axis=1), 'sum/1,1(<2)')
out(np.sum(np.sum(x, axis=2), axis=1), 'sum/2,1')

# np.sum(np.sum(x, axis=1), axis=2) XXX error

out(np.sum(np.sum(np.sum(x, axis=1), axis=1)), 'sum/1,1(<2),0')
out(np.sum(x), 'sum(/0,0,0)')

