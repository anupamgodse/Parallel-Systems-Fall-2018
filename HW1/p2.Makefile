#Single Author info:

#

#yjkamdar Yash J Kamdar

#

#

#Group info:

#

#angodse Anupam N Godse

#

#vphadke Vandan V Phadke

#

#Generate executable file

p2: p2.c
	mpicc -lm -O3 -o p2 p2.c p2_func.c



#Clean previous makes

clean:
	rm p2
