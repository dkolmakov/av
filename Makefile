APP=runner

CXX = g++
INCLUDES += -Iinclude/
CXXFLAGS = -Wall -std=c++11 -fpermissive -O3 

OBJDIR=./obj
BINDIR=./bin
LOGDIR=./log

SOURCES := $(wildcard *.cpp)
HEADERS := $(wildcard include/*.hpp)

OBJECTS_NONE := $(OBJDIR)/$(SOURCES:.cpp=.onone)
OBJECTS_SSE := $(OBJDIR)/$(SOURCES:.cpp=.osse)
OBJECTS_AVX := $(OBJDIR)/$(SOURCES:.cpp=.oavx)
OBJECTS_AVX512 := $(OBJDIR)/$(SOURCES:.cpp=.oavx512)

BINARIES := $(BINDIR)/$(APP)_none $(BINDIR)/$(APP)_sse $(BINDIR)/$(APP)_avx $(BINDIR)/$(APP)_avx512 

$(BINDIR):
	mkdir -p bin
	
$(OBJDIR):
	mkdir -p obj

$(LOGDIR):
	mkdir -p log

$(BINDIR)/$(APP)_none: $(OBJECTS_NONE) $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BINDIR)/$(APP)_sse: $(OBJECTS_SSE) $(HEADERS)
	$(CXX) $(CXXFLAGS) -msse4.1 $(OBJECTS_SSE) -o $@

$(BINDIR)/$(APP)_avx: $(OBJECTS_AVX) $(HEADERS)
	$(CXX) $(CXXFLAGS) -mavx $(OBJECTS_AVX) -o $@

$(BINDIR)/$(APP)_avx512: $(OBJECTS_AVX512) $(HEADERS)
	$(CXX) $(CXXFLAGS) -mavx512f $(OBJECTS_AVX512) -o $@

$(OBJDIR)/%.onone: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ 

$(OBJDIR)/%.osse: %.cpp
	$(CXX) $(CXXFLAGS) -msse4.1 $(INCLUDES) -c $< -o $@ 

$(OBJDIR)/%.oavx: %.cpp
	$(CXX) $(CXXFLAGS) -mavx $(INCLUDES) -c $< -o $@ 

$(OBJDIR)/%.oavx512: %.cpp
	$(CXX) $(CXXFLAGS) -mavx512f $(INCLUDES) -c $< -o $@ 

$(LOGDIR)/sum%.asm: $(BINDIR)/$(APP)%
	objdump -d $< | awk -v RS= '/<main>/' | awk -v RS="" -F '[ \t]+[a-z0-9]+:[ \t]+90[ \t]*nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop' '{print $$2}' > $@

asm: $(LOGDIR) $(LOGDIR)/sum_sse.asm $(LOGDIR)/sum_avx.asm $(LOGDIR)/sum_avx512.asm $(LOGDIR)/sum_none.asm 
	
all: $(OBJDIR) $(BINDIR) $(BINARIES) asm

clean:
	@rm -rvf $(OBJDIR)/* $(LOGDIR)/* 
	

