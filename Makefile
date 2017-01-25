CC = g++
CFLAGS = -Wall -O3 -fopenmp -Igzstream -Isrc -Isrc/models -std=c++0x
LDFLAGS = -lgomp -lgzstream -lz -lstdc++ -Lgzstream -larmadillo
OBJECTS = obj/common.o obj/corpus.o obj/model.o gzstream/gzstream.o obj/tsne.o obj/sptree.o obj/categoryTree.o
DEL_OBJECT =  obj/common.o obj/corpus.o obj/model.o gzstream/gzstream.o obj/categoryTree.o
MODELOBJECTS = obj/models/POP.o obj/models/WRMF.o obj/models/BPRMF.o obj/models/TimeBPR.o obj/models/BPRTMF.o obj/models/VBPR.o obj/models/TVBPR.o obj/models/TVBPRplus.o 

all: train

obj/common.o: src/common.hpp src/common.cpp Makefile
	$(CC) $(CFLAGS) -c src/common.cpp -o $@

obj/corpus.o: src/corpus.hpp src/corpus.cpp obj/common.o gzstream/gzstream.o Makefile
	$(CC) $(CFLAGS) -c src/corpus.cpp -o $@

obj/categoryTree.o: src/categoryTree.cpp src/categoryTree.hpp obj/common.o Makefile
	$(CC) $(CFLAGS) -c src/categoryTree.cpp -o $@
	

obj/model.o: src/model.hpp src/model.cpp obj/corpus.o obj/common.o obj/categoryTree.o Makefile
	$(CC) $(CFLAGS) -c src/model.cpp -o $@

obj/models/POP.o: src/models/POP.cpp src/models/POP.hpp obj/model.o obj/corpus.o obj/common.o Makefile
	$(CC) $(CFLAGS) -c src/models/POP.cpp -o $@

obj/models/WRMF.o: src/models/WRMF.cpp src/models/WRMF.hpp obj/model.o obj/corpus.o obj/common.o Makefile
	$(CC) $(CFLAGS) -c src/models/WRMF.cpp -o $@


obj/models/BPRMF.o: src/models/BPRMF.cpp src/models/BPRMF.hpp obj/model.o obj/corpus.o obj/common.o Makefile
	$(CC) $(CFLAGS) -c src/models/BPRMF.cpp -o $@

obj/models/VBPR.o: src/models/VBPR.cpp src/models/VBPR.hpp obj/models/BPRMF.o obj/model.o obj/corpus.o obj/common.o Makefile
	$(CC) $(CFLAGS) -c src/models/VBPR.cpp -o $@


obj/models/TimeBPR.o: src/models/TimeBPR.cpp src/models/TimeBPR.hpp obj/models/BPRMF.o obj/model.o obj/corpus.o obj/common.o Makefile
	$(CC) $(CFLAGS) -c src/models/TimeBPR.cpp -o $@

obj/models/BPRTMF.o: src/models/BPRTMF.cpp src/models/BPRTMF.hpp obj/models/TimeBPR.o obj/models/BPRMF.o obj/model.o obj/corpus.o obj/common.o Makefile
	$(CC) $(CFLAGS) -c src/models/BPRTMF.cpp -o $@

obj/models/TVBPR.o: src/models/TVBPR.cpp src/models/TVBPR.hpp obj/models/TimeBPR.o obj/models/BPRMF.o obj/model.o obj/corpus.o obj/common.o Makefile
	$(CC) $(CFLAGS) -c src/models/TVBPR.cpp -o $@

obj/models/TVBPRplus.o: src/models/TVBPRplus.cpp src/models/TVBPRplus.hpp obj/models/TVBPR.o obj/models/TimeBPR.o obj/models/BPRMF.o obj/model.o obj/corpus.o obj/common.o Makefile
	$(CC) $(CFLAGS) -c src/models/TVBPRplus.cpp -o $@


gzstream/gzstream.o:
	cd gzstream && make


train: src/main.cpp $(OBJECTS) $(MODELOBJECTS) Makefile
	$(CC) $(CFLAGS) -o train src/main.cpp $(OBJECTS) $(MODELOBJECTS) $(LDFLAGS)

clean:
	rm -rf $(DEL_OBJECTS) $(MODELOBJECTS) train
