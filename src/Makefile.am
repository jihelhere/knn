#AM_CPPFLAGS= -DNDEBUG
#AM_CXXFLAGS= -O6

bin_PROGRAMS = knn

knn_SOURCES = knn.cc Predictor.hh utils.h utils.c
knn_LDFLAGS = -pthread

if WANT_BOOST_THREAD
knn_LDFLAGS+= $(BOOST_LDFLAGS) $(BOOST_SYSTEM_LIB) $(BOOST_THREAD_LIB)
endif
