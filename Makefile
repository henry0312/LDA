.PHONY: all clean distclean


include config.mak

OBJS=$(SRCS:%.cpp=%.o)


include config.mak2

all: $(PROG)

$(PROG): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LIBS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(PROG) $(OBJS)

distclean: clean
	rm -f config.*
