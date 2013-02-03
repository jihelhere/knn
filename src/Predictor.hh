#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include <queue>

#include "Example.hh"
#include "ExampleMaker.hh"

#include "utils.h"

int processed_lines = 0;
int finished = 0;
threadns::mutex mutex_processed_lines;
threadns::mutex mutex_examples;
threadns::condition_variable cond_process;

namespace knn {

struct file_reader
{
  char * buffer;
  size_t buffer_size;

  file_reader() : buffer(NULL), buffer_size(0) {}

  ~file_reader() {free(buffer);}

  void
  process_file(FILE** fp, std::vector<char*>* lines)
  {
    while(0 < read_line(&buffer, &buffer_size, *fp))  {
      if(buffer[0] != '\n') {

        mutex_processed_lines.lock();
        lines->push_back(strdup(buffer));
        cond_process.notify_all();
        //        fprintf(stderr, "after notify, size lines: %ld\n", lines->size());
        mutex_processed_lines.unlock();
      }
      else break;

    }
    finished = 1;
    cond_process.notify_all();
  }
};


struct Predictor {

  enum distance_type {EUCLIDEAN, COSINE};

  int num_threads;
  std::vector<Example*> training_examples;
  unsigned k;
  distance_type dt;

  Predictor(int numthreads, const std::string& trainname, unsigned K, distance_type DT) :
      num_threads(numthreads), training_examples(), k(K), dt(DT)
  {
    load_train(trainname);
  }

  void load_train(const std::string& filename)
  {
    FILE* fp = fopen(filename.c_str(), "r");
    if(!fp) {
      fprintf(stderr, "ERROR: cannot load model from \"%s\"\n", filename.c_str());
      return;
    }


    file_reader fr;

    std::vector<char*> lines;
    std::vector<knn::ExampleMaker*> exampleMakers(num_threads, NULL);

    for(int i = 0; i < num_threads; ++i) {
      exampleMakers[i] = new knn::ExampleMaker(lines, training_examples);
    }

    processed_lines = 0;
    finished = 0;

    threadns::thread thread_read(&file_reader::process_file, &fr, &fp, &lines);

    for(int i = 0; i < num_threads; ++i) {
      exampleMakers[i]->start(&mutex_processed_lines, &processed_lines, &finished, &cond_process, &mutex_examples);
    }

    thread_read.join();

    for(auto i = exampleMakers.begin(); i != exampleMakers.end(); ++i)
      (*i)->join();

    fprintf(stderr, "%lu examples read\n", training_examples.size());

    for(auto& e : training_examples)
    {
      e->remove_noise(0.001);
    }

    fclose(fp);
  }

  std::string predict(Example& example) {

    example.remove_noise(0.001);

    std::priority_queue<Example> queue;

    threadns::thread tab[num_threads];

    for(int i = 0; i < num_threads; ++i) {
      tab[i] = threadns::thread((dt == EUCLIDEAN) ? &Example::compute_distances : &Example::compute_similarities,
                                &example,
                                training_examples,
                                i * training_examples.size() / num_threads,
                                (i+1)* training_examples.size() / num_threads);
    }

    for(int i = 0; i < num_threads; ++i)
    {
      tab[i].join();
    }

    // get the k nearest neighbours
    for (auto& e : training_examples)
    {
      if(queue.empty() || e->distance < queue.top().distance)
        queue.push(*e);

      if(queue.size() > this->k)
        queue.pop();
    }

    std::unordered_map<std::string,int> counts;
    while(!queue.empty())
    {
      fprintf(stdout, "size: %lu\tcategory: %s\tdistance: %f\n", queue.size(), queue.top().category.c_str(), queue.top().distance);
      counts[queue.top().category] += 1;
      queue.pop();
    }

    std::string res = "";
    int max = 0;

    for(auto& i : counts)
    {
      if(i.second > max)
      {
        res = i.first;
        max = i.second;
      }
    }


    return res;
  }
};

std::map<Predictor::distance_type, std::string> dt2string({{Predictor::EUCLIDEAN, "euclidean"}, {Predictor::COSINE, "cosine"}});
std::map<std::string, Predictor::distance_type> string2dt({{"euclidean", Predictor::EUCLIDEAN}, {"cosine", Predictor::COSINE}});

}
