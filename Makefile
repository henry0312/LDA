.PHONY: all clean distclean dep depend

include config.mak

OBJS=$(SRCS:%.cpp=%.o)

LDA_OBJS=$(LDA_SRCS:%.cpp=%.o)
HDPLDA_OBJS=$(HDPLDA_SRCS:%.cpp=%.o)

all: $(TOOLS)

lda: $(LDA_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LIBS) -o $@$(EXT) $^

hdplda: $(HDPLDA_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LIBS) -o $@$(EXT) $^

%.o: %.cpp .depend
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	$(RM) *.o *.exe $(TOOLS) .depend

distclean: clean
	$(RM) config.*

dep: .depend

depend: .depend

ifneq ($(wildcard .depend),)
include .depend
endif

#The dependency of each source file is solved automatically by follows.
.depend: config.mak
	@$(RM) .depend
	@$(foreach SRC, $(SRCS:%=$(SRCDIR)/%), $(CXX) $(SRC) $(CXXFLAGS) -g0 -MT $(SRC:$(SRCDIR)/%.cpp=%.o) -MM >> .depend;)

config.mak:
	./configure

