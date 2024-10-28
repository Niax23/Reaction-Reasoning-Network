import torch
import cProfile

def fun1():
	x = torch.randn(10, 1000).cuda()
	y = []
	for idx, t in enumerate(x):
		top = t.topk(10, largest=True, dim=-1)
		y.append(top.values)

	y = torch.cat(y, dim=0)
	z = y.topk(10, largest=True, dim=-1)
	return z

def fun2():
	x = torch.randn(10, 1000).cuda()
	z = x.flatten().topk(10, largest=True, dim=-1)
	return z

def main():
	for i in range(100):
		fun1()
		fun2()


if __name__ == '__main__':
	cProfile.run('main()')



