EXE=runner

CXX = g++
INCLUDES += -Iinclude/
CXXFLAGS = -Wall -std=c++11 -fpermissive $(optflags)

OBJDIR=./obj

SOURCES := $(wildcard *.cpp)
HEADERS := $(wildcard include/*.hpp) 
OBJECTS := $(OBJDIR)/$(SOURCES:.cpp=.o)

.PHONY: $(EXE)

$(EXE): $(OBJECTS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@

$(OBJDIR):
	mkdir -p obj

$(OBJDIR)/%.o: %.cpp
	@echo $@
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ 

all: $(OBJDIR) $(EXE)

clean:
	@rm -rvf $(EXE) $(OBJDIR)/*.o asm/*.asm 
	

