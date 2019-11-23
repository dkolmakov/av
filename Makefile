GENERAL=./bin/runner

CXX = g++
INCLUDES += -Iinclude/
CXXFLAGS = -Wall -std=c++11 -fpermissive -O3

OBJDIR=./obj
BINDIR=./bin

SOURCES := $(wildcard *.cpp)
HEADERS := $(wildcard include/*.hpp)

OBJECTS := $(OBJDIR)/$(SOURCES:.cpp=.o)
OBJECTS_SSE := $(OBJDIR)/$(SOURCES:.cpp=.osse)
OBJECTS_AVX := $(OBJDIR)/$(SOURCES:.cpp=.oavx)
OBJECTS_AVX512 := $(OBJDIR)/$(SOURCES:.cpp=.oavx512)

$(GENERAL): $(OBJECTS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@

$(GENERAL)_sse: $(OBJECTS_SSE) $(HEADERS)
	$(CXX) $(CXXFLAGS) -msse4.1 $(OBJECTS_SSE) -o $@

$(GENERAL)_avx: $(OBJECTS_AVX) $(HEADERS)
	$(CXX) $(CXXFLAGS) -mavx $(OBJECTS_AVX) -o $@

$(GENERAL)_avx512: $(OBJECTS_AVX512) $(HEADERS)
	$(CXX) $(CXXFLAGS) -mavx512f $(OBJECTS_AVX512) -o $@

$(BINDIR):
	mkdir -p bin
	
$(OBJDIR):
	mkdir -p obj

$(OBJDIR)/%.o: %.cpp
	@echo $@
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ 

$(OBJDIR)/%.osse: %.cpp
	@echo $@
	$(CXX) $(CXXFLAGS) -msse4.1 $(INCLUDES) -c $< -o $@ 

$(OBJDIR)/%.oavx: %.cpp
	@echo $@
	$(CXX) $(CXXFLAGS) -mavx $(INCLUDES) -c $< -o $@ 

$(OBJDIR)/%.oavx512: %.cpp
	@echo $@
	$(CXX) $(CXXFLAGS) -mavx512f $(INCLUDES) -c $< -o $@ 

all: $(OBJDIR) $(BINDIR) $(GENERAL) $(GENERAL)_sse $(GENERAL)_avx $(GENERAL)_avx512

clean:
	@rm -rvf $(GENERAL)* $(OBJDIR)/*.o $(OBJDIR)/*.osse $(OBJDIR)/*.oavx 
	

