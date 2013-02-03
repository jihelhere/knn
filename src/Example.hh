#pragma once

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <cassert>
#include <unordered_map>

#ifdef USE_BOOST_THREAD
#include <boost/thread.hpp>
namespace threadns = boost;
#else
#include <thread>
namespace threadns = std;
#endif

int counter = 0;
std::unordered_map<std::string,int> string_map;
std::unordered_map<int,double> count_map;
int count_map_counter;

threadns::mutex mutex_string_map;

typedef threadns::unique_lock<threadns::mutex> lock_type;

namespace knn {

struct Feature {

  unsigned id;
  double value;
  Feature(int _id, double _value) : id(_id), value(_value) {};

  inline bool operator<(const Feature &peer) const {
    return id < peer.id;
  }
};

struct Example {

  std::string id;
  std::string category;
  std::vector<Feature> features;
  double distance;

  Example() : id(), category(), features(), distance(0.0) {
  }


  Example(char* input, bool normalise, bool add_features)
      : id(), category(), features(), distance(0.0) {
    load(input, normalise, add_features);
  }

  // load an example from a line 'category feature_id:value .... feature_id:value' # comment
  // no error checking
  void load(char* line, bool normalise, bool add_features) {
    char * input = line;
    char *token = NULL;

    double norm = 0;


    //    fprintf(stderr, "%s", line);

    //TODO: uncomment this
    // char *comment = strchr(input, '#');
    // if(comment != NULL) *comment = '\0'; // skip comments

    token =  strsep(&input, " \t"); // read id

    id = token;

    token = strsep(&input, " \t"); // read category

    category = token;

    //    fprintf(stderr, "loss: %g\n", loss);
    for(;(token = strsep(&input, " \t\n")) && *token != '\0' ;) {
      //      fprintf(stderr, "token: %s\n", token);
      char* value = strrchr(token, ':');

      *value = '\0';

      if(strlen(token) < 5)
        continue;
      token[5] = '\0';

      double value_as_double = strtod(value + 1, NULL);

      if(normalise)
        norm += value_as_double * value_as_double;

      int feature_id = -1;

      lock_type lock2(mutex_string_map);
      if (add_features || string_map.count(token))
      {
        auto resf = string_map.insert(std::make_pair(token, counter));
        if(resf.second)
          ++counter;

        feature_id = resf.first->second;
      }

      if(add_features)
      {
        count_map[feature_id] += int(value_as_double);
        count_map_counter += int(value_as_double);
      }
      lock2.unlock();

      if(feature_id != -1)
        features.emplace_back(feature_id, value_as_double);
    }

    //    fprintf(stderr, "line loaded\n");

    if(normalise)
    {
      norm = std::sqrt(norm);

      for (auto& f : features)
      {
        f.value /= norm;
      }
    }

    std::sort(features.begin(), features.end());

    //    fprintf(stderr, "size: %d\n", features.size());


  }


  void remove_noise(double threshold)
  {
    features.erase
        (std::remove_if(features.begin(), features.end(),
                        [&](Feature& f)
                        {
                          return double(count_map[f.id]) / count_map_counter < threshold;
                        }
                        ),
         features.end());
  }


  void compute_distance( Example& other) {
    other.distance = 0;

    //    fprintf(stderr, "DISTANCE\n");

    auto i = this->features.begin(), j = other.features.begin();

    while(i != this->features.end() && j != other.features.end()) {

      //      fprintf(stderr,"%d:%f\t%d:%f\n", i->id, i->value, j->id, j->value);

      if(i->id < j->id)
      {
        other.distance += i->value * i->value;
        ++i;
      }
      else if (j->id < i->id)
      {
        other.distance += j->value * j->value;
        ++j;
      }
      else // equal
      {
        other.distance += (i->value - j->value) * (i->value - j->value);
        ++i; ++j;
      }
    }

    while(i != this->features.end()) { other.distance += i->value * i->value; ++i; }
    while(j != other.features.end()) { other.distance += j->value * j->value; ++j; }
  }

  void compute_distances(std::vector<Example*>& example, int begin, int end)
  {
    for (int i = begin; i < end; ++i)
    {
      compute_distance(*example[i]);
    }
  }

  void compute_similarity( Example& other) {
    other.distance = 0;

    //    fprintf(stderr, "DISTANCE\n");

    double magthis = 0 ;
    double magother = 0;


    auto i = this->features.begin(), j = other.features.begin();

    while(i != this->features.end() && j != other.features.end()) {

      //      fprintf(stderr,"%d:%f\t%d:%f\n", i->id, i->value, j->id, j->value);

      if(i->id < j->id)
      {
        magthis += i->value * i->value;
        ++i;
      }
      else if (j->id < i->id)
      {
        magother += j->value * j->value;
        ++j;
      }
      else // equal
      {
        other.distance += i->value * j->value;
        magthis += i->value * i->value;
        magother += j->value * j->value;
        ++i; ++j;
      }
    }

    while(i != this->features.end()) { magthis  += i->value * i->value; ++i; }
    while(j != other.features.end()) { magother += j->value * j->value; ++j; }

    other.distance /= std::sqrt(magthis) * std::sqrt(magother);
    other.distance = 1 - other.distance;
  }

  void compute_similarities(std::vector<Example*>& example, int begin, int end)
  {
    for (int i = begin; i < end; ++i)
    {
      compute_similarity(*example[i]);
    }
  }


  inline bool operator<(const Example& o) const
  {
    return distance < o.distance;
  }

};

}
