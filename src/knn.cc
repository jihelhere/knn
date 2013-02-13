#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "Predictor.hh"

#include <getopt.h>

#define NUM_THREADS 1
#define NUM_NEIGHBOURS 10
#define DISTANCE "cosine"



void  print_help_message(char *program_name)
{
fprintf(stderr, "%s usage: %s [options] < file\n", program_name, program_name);
 fprintf(stderr, "OPTIONS :\n");
 fprintf(stderr, "      --train,-t             : example file\n");
 fprintf(stderr, "      --threads,-j           : nb of threads (default is %d)\n", NUM_THREADS);
 fprintf(stderr, "      --k,-k                 : nb of neighbours (default is %d)\n", NUM_NEIGHBOURS);
 fprintf(stderr, "      --distance,-d          : type of distance euclidean or cosine (default is %s)\n", DISTANCE);
 fprintf(stderr, "      --eval,-e              : evaluation mode\n");
 fprintf(stderr, "      -help,-h               : print this message\n");
}


int main(int argc, char** argv) {
  char * train = NULL;
  int threads = NUM_THREADS;
  int k = NUM_NEIGHBOURS;
  bool eval = false;

  std::string distance = DISTANCE;

  // read the commandline
  int c;
  while(1) {

    static struct option long_options[] =
      {
        /* These options don't set a flag.
           We distinguish them by their indices. */
        {"help",     no_argument,             0, 'h'},
        {"eval",     no_argument,             0, 'e'},
        {"train",    required_argument,       0, 't'},
        {"k",        required_argument,       0, 'k'},
        {"threads",  required_argument,       0, 'j'},
        {"distance", required_argument,       0, 'd'},
        {0, 0, 0, 0}
      };

    // int to store arg position
    int option_index = 0;

    c = getopt_long (argc, argv, "j:t:k:hed:", long_options, &option_index);

    // Detect the end of the options
    if (c == -1)
      break;

    switch (c)
      {
      case 0:
        // If this option set a flag, do nothing else now.
        if (long_options[option_index].flag != 0)
          break;
        fprintf(stderr, "option %s", long_options[option_index].name);
        if (optarg)
          fprintf(stderr, " with arg %s", optarg);
        fprintf (stderr, "\n");
        break;


      case 'h':
        print_help_message(argv[0]);
        exit(0);

      case 'e':
        eval = true;
        break;

      case 't':
        fprintf (stderr, "train filename: %s\n", optarg);
        train = optarg;
        break;

      case 'k':
        fprintf (stderr, "number of neighbours to consider: %s\n", optarg);
        k = atoi(optarg);
        break;

      case 'j':
        fprintf (stderr, "number of threads: %s\n", optarg);
        threads = atoi(optarg);
        break;


      case 'd':
        fprintf(stderr, "distance: %s\n", optarg);
        distance = optarg;
        break;
      case '?':
        // getopt_long already printed an error message.
        break;

      default:
        abort ();
      }

  }

  if(train == NULL || threads <= 0 || k < 0) {
    print_help_message(argv[0]);
    return 1;
  }

  knn::Predictor<knn::ZNormaliser>
      predictor(threads, train, k, knn::string2dt.at(distance));

  fprintf(stderr, "\n\nTraining examples loaded\n\n");


  char* buffer = NULL;
  size_t buffer_length = 0;
  ssize_t length = 0;

  int total = 0;
  int correct = 0;

  while(0 <= (length = read_line(&buffer, &buffer_length, stdin))) {

    //    fprintf(stderr, "example: %d", total);

    knn::Example example(buffer, true, false);
    //knn::Example example(buffer, false, false);
    example.remove_noise(0.0001);
    predictor.normaliser.normalise(&example);

    std::string ref_example = example.category;
    std::string hyp_example = predictor.predict(example);
    fprintf(stdout, "%s %s\n", example.id.c_str(), hyp_example.c_str());
    ++total;
    if(eval)
    {
      if(ref_example == hyp_example)
        ++correct;
      fprintf(stderr, "correct: %d\ttotal: %d\taccuracy: %f\n", correct, total, double(correct)/total);
    }
  }
  if(eval)
  {
    fprintf(stderr, "correct: %d\ttotal: %d\taccuracy: %f\n", correct, total, double(correct)/total);
  }

  return 0;
}
