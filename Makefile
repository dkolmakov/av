APP=runner

CXX = g++
INCLUDES += -Iinclude/
CXXFLAGS = -Wall -std=c++11 -fpermissive -O3
#-fopt-info-vec-all

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

SUM_SIMPLE_ASM := $(subst $(BINDIR)/$(APP),$(LOGDIR)/sum_simple,$(BINARIES))
SUM_UNROLL_ASM := $(subst $(BINDIR)/$(APP),$(LOGDIR)/sum_unroll,$(BINARIES))
SUM_CHUNKED_ASM := $(subst $(BINDIR)/$(APP),$(LOGDIR)/sum_chunked,$(BINARIES))
SUM_MANUAL_ASM := $(subst $(BINDIR)/$(APP),$(LOGDIR)/sum_man,$(BINARIES))

MUL_SIMPLE_ASM := $(subst $(BINDIR)/$(APP),$(LOGDIR)/mul_simple,$(BINARIES))
MUL_UNROLL_ASM := $(subst $(BINDIR)/$(APP),$(LOGDIR)/mul_unroll,$(BINARIES))
MUL_MANUAL_ASM := $(subst $(BINDIR)/$(APP),$(LOGDIR)/mul_man,$(BINARIES))
MUL_ADVANCED_ASM := $(subst $(BINDIR)/$(APP),$(LOGDIR)/mul_adv,$(BINARIES))

all: $(OBJDIR) $(BINDIR) $(BINARIES) asm

clean:
	@rm -rvf $(OBJDIR)/* $(LOGDIR)/* $(BINARIES)
	
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

	
$(LOGDIR)/sum_simple%: $(BINDIR)/$(APP)%
	objdump -d $< | awk -v RS= '/<main>/' | awk -v RS="" -F '[ \t]+[a-z0-9]+:[ \t]+90[ \t]*nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop' '{print $$2}' > $@

$(LOGDIR)/sum_unroll%: $(BINDIR)/$(APP)%
	objdump -d $< | awk -v RS= '/<main>/' | awk -v RS="" -F '[ \t]+[a-z0-9]+:[ \t]+90[ \t]*nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop' '{print $$4}' > $@

$(LOGDIR)/sum_chunked%: $(BINDIR)/$(APP)%
	objdump -d $< | awk -v RS= '/<main>/' | awk -v RS="" -F '[ \t]+[a-z0-9]+:[ \t]+90[ \t]*nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop' '{print $$6}' > $@
	
$(LOGDIR)/sum_man%: $(BINDIR)/$(APP)%
	objdump -d $< | awk -v RS= '/<main>/' | awk -v RS="" -F '[ \t]+[a-z0-9]+:[ \t]+90[ \t]*nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop' '{print $$8}' > $@
	
$(LOGDIR)/mul_simple%: $(BINDIR)/$(APP)%
	objdump -d $< | awk -v RS= '/<main>/' | awk -v RS="" -F '[ \t]+[a-z0-9]+:[ \t]+90[ \t]*nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop' '{print $$10}' > $@

$(LOGDIR)/mul_unroll%: $(BINDIR)/$(APP)%
	objdump -d $< | awk -v RS= '/<main>/' | awk -v RS="" -F '[ \t]+[a-z0-9]+:[ \t]+90[ \t]*nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop' '{print $$12}' > $@

$(LOGDIR)/mul_man%: $(BINDIR)/$(APP)%
	objdump -d $< | awk -v RS= '/<main>/' | awk -v RS="" -F '[ \t]+[a-z0-9]+:[ \t]+90[ \t]*nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop' '{print $$14}' > $@

$(LOGDIR)/mul_adv%: $(BINDIR)/$(APP)%
	objdump -d $< | awk -v RS= '/<main>/' | awk -v RS="" -F '[ \t]+[a-z0-9]+:[ \t]+90[ \t]*nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop\n[ \t]+[a-z0-9]+:[ \t]+90[ \t]+nop' '{print $$16}' > $@
	
asm: $(LOGDIR) $(SUM_SIMPLE_ASM) $(SUM_UNROLL_ASM) $(SUM_CHUNKED_ASM) $(SUM_MANUAL_ASM) $(MUL_SIMPLE_ASM) $(MUL_UNROLL_ASM) $(MUL_MANUAL_ASM) $(MUL_ADVANCED_ASM)

	

