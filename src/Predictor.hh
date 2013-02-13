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
enum distance_type {EUCLIDEAN, COSINE};


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


struct MaxMinNormaliser
{
  std::vector<double> mins;
  std::vector<double> maxs;

  MaxMinNormaliser() : mins(), maxs() {};

  void init(std::vector<Example*> training_examples)
  {
    for(const auto& e: training_examples)
    {
      for(const auto& f : e->features)
      {
        if(mins.size() <= f.id)
          mins.resize(f.id+1, std::numeric_limits<double>::infinity());
        if(maxs.size() <= f.id)
          maxs.resize(f.id+1, - std::numeric_limits<double>::infinity());

        if(f.value < mins[f.id])
          mins[f.id] = f.value;
        if(f.value > maxs[f.id])
          maxs[f.id] = f.value;
      }
    }
  }

  void normalise(Example* e) const
  {
    for(auto& f : e->features)
    {
      f.value = (f.value - this->mins[f.id]) / (this->maxs[f.id] - this->mins[f.id]);
    }
  }


};


struct ZNormaliser
{
  std::vector<double> means;
  std::vector<double> deviations;

  ZNormaliser() : means(), deviations() {};

  void init(std::vector<Example*> training_examples)
  {
    for(const auto& e: training_examples)
    {
      for(const auto& f : e->features)
      {
        if(means.size() <= f.id) means.resize(f.id+1, 0);
        means[f.id] += f.value;
      }
    }
    for (size_t i = 0; i < means.size(); ++i)
    {
      means[i] /= training_examples.size();
    }
    deviations.resize(means.size());

    for(const auto& e: training_examples)
    {
      for(const auto& f : e->features)
      {
        deviations[f.id] = (f.value - means[f.id]) * (f.value - means[f.id]);
      }
    }

    for (size_t i = 0; i < deviations.size(); ++i)
    {
      deviations[i] = std::sqrt( deviations[i] / (deviations.size() - 1));
    }




  }

  void normalise(Example* e) const
  {
    for(auto& f : e->features)
    {
      f.value = (f.value - this->means[f.id]) / this->deviations[f.id];
    }
  }


};


template<class Normaliser>
struct Predictor {

  int num_threads;
  std::vector<Example*> training_examples;
  unsigned k;
  distance_type dt;
  Normaliser normaliser;


  Predictor(int numthreads, const std::string& trainname, unsigned K, distance_type DT) :
      num_threads(numthreads), training_examples(), k(K), dt(DT), normaliser()
  {
    load_train(trainname);
    normaliser.init(training_examples);
    std::for_each(training_examples.begin(), training_examples.end(),
                  [&](Example *e)
                  {
                    normaliser.normalise(e);
                  }
                  );
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
      e->remove_noise(0.0001);
    }

    fclose(fp);
  }

  std::string predict(Example& example) {

    std::priority_queue<Example> queue;

    threadns::thread tab[num_threads];

    for(int i = 0; i < num_threads; ++i) {
      tab[i] = threadns::thread((dt == EUCLIDEAN) ?
                                &Example::compute_distances :
                                &Example::compute_similarities,
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
      // fprintf(stdout, "size: %lu\tcategory: %s\tdistance: %f\n", queue.size(), queue.top().category.c_str(), queue.top().distance);
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

std::map<distance_type, std::string>
dt2string(
    {
      {EUCLIDEAN, "euclidean"},
      {COSINE, "cosine"}
    });

std::map<std::string, distance_type>
string2dt(
    {
      {"euclidean", EUCLIDEAN},
      {"cosine", COSINE}
    });
}
