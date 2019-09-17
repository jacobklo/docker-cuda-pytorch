import torch, timeit, functools
def test(a,b):
	return a.matmul(b)

A = torch.randn(15000,15000)
B = torch.randn(15000,15000)
Ag = A.cuda()
Bg = B.cuda()
cpu_timer = timeit.Timer(functools.partial(test,A,B))
cpu_time = cpu_timer.timeit(1)
gpu_timer = timeit.Timer(functools.partial(test,Ag,Bg))
gpu_time = gpu_timer.timeit(1)


print(cpu_time)
# 73.34245827499944
print(gpu_time)
# 0.3741293080001924

