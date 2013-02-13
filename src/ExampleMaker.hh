#ifndef _EXAMPLEMAKER_HH_
#define _EXAMPLEMAKER_HH_

#include <vector>
#include <unordered_map>

#include "Example.hh"

namespace knn {

  struct ExampleMaker
  {

    threadns::thread my_thread;

    std::vector<char*>& lines;
    std::vector<Example*>& examples;

    ExampleMaker(std::vector<char*> &l, std::vector<Example*>& e)
      : my_thread(), lines(l), examples(e) {};

    ~ExampleMaker() {}

    void join() {
      my_thread.join();
    }

    void create_examples(threadns::mutex* mlines, int* processed_lines, int* finished, threadns::condition_variable* cond_process, threadns::mutex* mutex_examples)
    {
      while(1) {
	int index;
	char * string;

        lock_type lock(*mlines);

        while(((unsigned) *processed_lines == lines.size()) && !*finished) {
#ifdef USE_BOOST_THREAD
	  //cond_process->wait(lock); // no deadlock with boost implementation  ?
          cond_process->timed_wait(lock, boost::posix_time::milliseconds(10));
#else
	  // prevent deadlock with posix implementation
	 cond_process->wait_for(lock, std::chrono::duration<int,std::milli>(10));
#endif
        }

	if(*finished && ((unsigned) *processed_lines == lines.size())) { lock.unlock(); break;}

	index = (*processed_lines)++;
	string = lines[index];

        lock.unlock();

        Example * e = new Example(string, true, true);
        //	Example * e = new Example(string, false, true);

        lock_type lock2(*mutex_examples);
        if (examples.size() < lines.size())
          examples.resize(lines.size(),NULL);
	examples[index] = e;
        if(index % 1000 == 0)
        fprintf(stderr, "%d examples read\r", index);
        lock2.unlock();

        delete lines[index];


        //        fprintf(stderr, "after resize\n");


      }
    }

    void start(threadns::mutex* mlines, int* processed_lines, int* finished, threadns::condition_variable* cond_process, threadns::mutex* mutex_examples)
    {
      my_thread = threadns::thread(&ExampleMaker::create_examples, this,
				   mlines, processed_lines, finished, cond_process,
                                   mutex_examples);
    }


  };
}



#endif
