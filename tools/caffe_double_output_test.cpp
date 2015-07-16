#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <sstream> 
#include <climits> 

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

const int MIN_PROB = -100;

DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_int32(ignore, 1,
    "Run whether to ignore the ignore label.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

const char *object_name[] = {"animal", "plant", "food", "traffic", "landscape", "portrait", "others"};
std::vector<std::string> object_name_dict(object_name, object_name + 7);

const char *scene_name[] = {"indoor", "outdoor", "others"};
std::vector<std::string> scene_name_dict(scene_name, scene_name + 3);

void print_test_score(const std::string& str, const int gt_label, const int pred_label, vector<float> score, bool isobject){
  std::vector<std::string> dict;
  if (isobject){
    dict = object_name_dict;
  } else {
    dict = scene_name_dict;
  }
  std::cout << str << std::endl;
  std::cout << "------TRUE LABEL: " << dict[gt_label] << " \t PREDICT LABEL: " \
      << dict[pred_label] << "----------" << std::endl;
  for (int i = 0; i < score.size(); i++){
    std::cout << dict[i] << " " << score[i] << std::endl;
  }
}

// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Parse source file from prototxt
  std::ifstream infile_model(FLAGS_model.c_str());
  std::string temp_line;
  std::string buffer;
  std::string data_dir;
  while (std::getline(infile_model, temp_line)){
    std::stringstream ss(temp_line);
    ss >> buffer;
    if (buffer.compare(0, 6, "source") == 0){
      ss >> data_dir;
      break;
    }
  }
  CHECK_GT(data_dir.size(), 0) << "No data path found in prototxt.";

  // Parse interation number from data source
  LOG(INFO) << "Opening file " << data_dir;
  data_dir = data_dir.substr(1, data_dir.size() - 2);
  std::ifstream infile(data_dir.c_str());
  vector<std::pair<std::string, std::pair<int, int> > > lines;
  std::string filename;
  int label;
  int label_second;
  while (infile >> filename >> label >> label_second) {
    lines.push_back(std::make_pair(filename, std::make_pair(label, label_second)));
  }
  int iterations = lines.size();

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << iterations << " iterations.";

  vector<Blob<float>* > bottom_vec;
  vector<float> test_score;

  float accuracy_object = 0;
  float count_object = 0;
  float accuracy_scene = 0;
  float count_scene = 0;
  vector<float> accuracy_object_vector (7, 0.0);
  vector<float> count_object_vector (7, 0.0);
  vector<float> accuracy_scene_vector (3, 0.0);
  vector<float> count_scene_vector (3, 0.0);

  for (int i = 0; i < iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(bottom_vec, &iter_loss);
    
    test_score.clear();
    const float* prob_object_vec = result[0]->cpu_data();
    float max_prob = MIN_PROB;
    int max_label = -1;
    for (int j = 0; j < result[0]->count(); j++){
      const float score = prob_object_vec[j];
      test_score.push_back(score);
      if (!isnan(score)){
        if (score > max_prob){
          max_prob = score;
          max_label = j;
        }
      }
    }
    if ( (FLAGS_ignore) & (lines[i].second.first != -1) ) {
      count_object += 1;      
      count_object_vector[lines[i].second.first] += 1;
      if (max_label == lines[i].second.first){
        accuracy_object += 1;       
        accuracy_object_vector[lines[i].second.first] += 1;       
      } else {
        print_test_score(lines[i].first, lines[i].second.first, max_label, test_score, true);
      }  
    }

    test_score.clear();
    const float* prob_scene_vec = result[1]->cpu_data();
    max_prob = MIN_PROB;
    max_label = -1;
    for (int j = 0; j < result[1]->count(); j++){
      const float score = prob_scene_vec[j];
      test_score.push_back(score);
      if (!isnan(score)){
        if (score > max_prob){
          max_prob = score;
          max_label = j;
        }
      }
    }
    if ( (FLAGS_ignore) & (lines[i].second.second != -1) ) {
      count_scene += 1;      
      count_scene_vector[lines[i].second.second] += 1;
      if (max_label == lines[i].second.second){
        accuracy_scene += 1;       
        accuracy_scene_vector[lines[i].second.second] += 1;       
      } else {
        print_test_score(lines[i].first, lines[i].second.second, max_label, test_score, false);
      }  
    }
  }

  LOG(INFO) << "Object accuracy: " << accuracy_object / count_object * 1.0 << " " << accuracy_object << " " << count_object;
  LOG(INFO) << "Scene accuracy: " << accuracy_scene / count_scene * 1.0 << " " << accuracy_scene << " " << count_scene;
  LOG(INFO) << " " ;
  for (int i = 0; i < 6; i++){
    LOG(INFO) << object_name_dict[i] << " accuracy: " << accuracy_object_vector[i] / 
        count_object_vector[i] * 1.0 << " " << accuracy_object_vector[i] << " " << count_object_vector[i];
  }
  for (int i = 0; i < 2; i++){
    LOG(INFO) << scene_name_dict[i] << " accuracy: " << accuracy_scene_vector[i] / 
        count_scene_vector[i] * 1.0 << " " << accuracy_scene_vector[i] << " " << count_scene_vector[i];
  }

  return 0;
}
RegisterBrewFunction(test);


int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  test            score a model\n"
      );
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
    return GetBrewFunction(caffe::string(argv[1]))();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
