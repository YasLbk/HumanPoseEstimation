#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
using namespace dnn;
/**
 * @brief IF DEBUG flag is true display the intermidiate results
 *
 */
#define DEBUG false

/**
 * @brief
 *
 * @param filename
 * @return true
 * @return false
 */
bool is_file_exist(const char *filename) {
  ifstream infile(filename);
  return infile.good();
}

/**
 * @brief Get the pose info object
 *
 * @param body_parts
 * @param pose_pairs
 * @return int
 */
int get_pose_info(map<string, int> &body_parts,
                  vector<pair<int, int>> &pose_pairs) {
  int num_points = 18;
  body_parts["Nose"] = 0;
  body_parts["Neck"] = 1;
  body_parts["RShoulder"] = 2;
  body_parts["RElbow"] = 3;
  body_parts["RWrist"] = 4;
  body_parts["LShoulder"] = 5;
  body_parts["LElbow"] = 6;
  body_parts["LWrist"] = 7;
  body_parts["RHip"] = 8;
  body_parts["RKnee"] = 9;
  body_parts["RAnkle"] = 10;
  body_parts["LHip"] = 11;
  body_parts["LKnee"] = 12;
  body_parts["LAnkle"] = 13;
  body_parts["REye"] = 14;
  body_parts["LEye"] = 15;
  body_parts["REar"] = 16;
  body_parts["LEar"] = 17;
  body_parts["Background"] = 18;

  pose_pairs.resize(18);
  pose_pairs[0] = make_pair(body_parts["Neck"], body_parts["RShoulder"]);
  pose_pairs[1] = make_pair(body_parts["Neck"], body_parts["LShoulder"]);
  pose_pairs[2] = make_pair(body_parts["RShoulder"], body_parts["RElbow"]);
  pose_pairs[3] = make_pair(body_parts["RElbow"], body_parts["RWrist"]);
  pose_pairs[4] = make_pair(body_parts["LShoulder"], body_parts["LElbow"]);
  pose_pairs[5] = make_pair(body_parts["LElbow"], body_parts["LWrist"]);
  pose_pairs[6] = make_pair(body_parts["Neck"], body_parts["RHip"]);
  pose_pairs[7] = make_pair(body_parts["RHip"], body_parts["RKnee"]);
  pose_pairs[8] = make_pair(body_parts["RKnee"], body_parts["RAnkle"]);
  pose_pairs[9] = make_pair(body_parts["Neck"], body_parts["LHip"]);
  pose_pairs[10] = make_pair(body_parts["LHip"], body_parts["LKnee"]);
  pose_pairs[11] = make_pair(body_parts["LKnee"], body_parts["LAnkle"]);
  pose_pairs[12] = make_pair(body_parts["Neck"], body_parts["Nose"]);
  pose_pairs[13] = make_pair(body_parts["Nose"], body_parts["REye"]);
  pose_pairs[14] = make_pair(body_parts["REye"], body_parts["REar"]);
  pose_pairs[15] = make_pair(body_parts["Nose"], body_parts["LEye"]);
  pose_pairs[16] = make_pair(body_parts["LEye"], body_parts["LEar"]);
  pose_pairs[17] = make_pair(body_parts["RHip"], body_parts["LHip"]);

  return num_points;
}

/**
 * @brief
 *
 * @param c
 * @param num
 * @return int
 */
int keyboard_mapping(char c, int num) {
  map<char, int> KEYBOARD_MAPPING;
  KEYBOARD_MAPPING['w'] = 0;   // Nose
  KEYBOARD_MAPPING['e'] = 1;   // Neck
  KEYBOARD_MAPPING['r'] = 2;   // RShoulder
  KEYBOARD_MAPPING['t'] = 3;   // RElbow
  KEYBOARD_MAPPING['y'] = 4;   // RWrist
  KEYBOARD_MAPPING['u'] = 5;   // LShoulder
  KEYBOARD_MAPPING['i'] = 6;   // LElbow
  KEYBOARD_MAPPING['o'] = 7;   // LWrist
  KEYBOARD_MAPPING['p'] = 8;   // RHip
  KEYBOARD_MAPPING['a'] = 9;   // RKnee
  KEYBOARD_MAPPING['s'] = 10;  // RAnkle
  KEYBOARD_MAPPING['d'] = 11;  // LHip
  KEYBOARD_MAPPING['f'] = 12;  // LKnee
  KEYBOARD_MAPPING['g'] = 13;  // LAnkle
  KEYBOARD_MAPPING['h'] = 14;  // REye
  KEYBOARD_MAPPING['j'] = 15;  // LEye
  KEYBOARD_MAPPING['k'] = 16;  // REar
  KEYBOARD_MAPPING['l'] = 17;  // LEar
  KEYBOARD_MAPPING['z'] = 18;  // Background
  KEYBOARD_MAPPING['x'] = 19;  // skeleton
  KEYBOARD_MAPPING['c'] = 20;  // skeleton
  KEYBOARD_MAPPING['v'] = 21;  // skeleton
  KEYBOARD_MAPPING['b'] = 22;  // skeleton
  KEYBOARD_MAPPING['n'] = 23;  // skeleton
  KEYBOARD_MAPPING['m'] = 24;  // skeleton

  if (KEYBOARD_MAPPING.find(c) != KEYBOARD_MAPPING.end()) {
    return KEYBOARD_MAPPING[c];
  } else {
    return num;
  }
}

/**
 * @brief Display the image
 *
 * @param img: input opencv image
 * @param title: title string of the window
 */
void display_image(Mat &img, string title = "Display_Window") {
  namedWindow(title, WINDOW_AUTOSIZE);
  imshow(title, img);
}

void print_dnn_net(Net &net, int batch_size = 1, int channels = 3,
                   int WINSIZE = 368) {
  MatShape ms1 = {batch_size, channels, WINSIZE, WINSIZE};
  vector<String> lnames = net.getLayerNames();
  for (size_t i = 1; i < lnames.size() + 1; i++) {
    Ptr<Layer> lyr = net.getLayer((unsigned)i);
    vector<MatShape> in, out;
    net.getLayerShapes(ms1, i, in, out);
    cout << lyr->name.c_str() << "  " << lyr->type.c_str();
    for (auto j : in) cout << "i" << Mat(j).t() << "\t";
    for (auto j : out) cout << "o" << Mat(j).t() << "\t";
    for (auto b :
         lyr->blobs) {  // what the net trains on, e.g. weights and bias
      cout << "b[" << b.size[0];
      for (size_t d = 1; d < b.dims; d++) cout << ", " << b.size[d];
      cout << "]  ";
    }
    cout << endl;
  }
}
/*---------------------------------------------------------------*/
/**
 * @brief load the tf model
 *
 * @param dp_name: name of model
 * @param hpe_net: net file
 */
void load_trained_model(string dp_name, Net &hpe_net) {
  hpe_net = readNetFromTensorflow(dp_name);
  CV_Assert(hpe_net.empty() == false);
  // print_dnn_net(hpe_net);
}

/**
 * @brief forwarding the netfile
 *
 * @param input_image :  input image
 * @param hpe_net: net file
 * @param output_dnn : output matrix after forwarding net
 * @param inuput_height
 * @param input_width
 * 
 */
void dnn_forward_inference(Mat &input_image, Net &hpe_net, Mat &output_dnn,
                           double input_height = 368,
                           double input_width = 368) {
  Mat input_blob =
      blobFromImage(input_image, 1.0, Size(input_width, input_height),
                    Scalar(127.5, 127.5, 127.5), false, false);
  CV_Assert(input_blob.dims == 4 && input_blob.data != 0);
  CV_Assert(input_blob.depth() == CV_8U || input_blob.depth() == CV_32F);
  hpe_net.setInput(input_blob);
  output_dnn = hpe_net.forward();
}
/**
 * @brief forwarding the netfile
 *
 * @param input_img : input image
 * @param output_dnn : output matrix after forwarding net
 * @param select_map : selected bodypart
 * @param result : output
 * 
 */
Mat draw_heatmap(Mat &input_img, Mat &output_dnn, int select_map, Mat &result) {
  Mat normalized_heatmap, gray_heatmap, colored_map;

  // Extract the heatmap corresponding to select_map from the output_dnn (its
  // size would be 46x46). Let’s name it result_heatmap_overlay.
  Mat result_heatmap_overlay(46, 46, CV_32F, output_dnn.ptr(0, select_map));
  CV_Assert(result_heatmap_overlay.data);

  normalize(result_heatmap_overlay, colored_map, 0.0, 255.0, NORM_MINMAX,
            CV_8UC1);
  applyColorMap(colored_map, colored_map, COLORMAP_JET);

  CV_Assert(colored_map.data);
  CV_Assert(result_heatmap_overlay.data);
  resize(colored_map, result_heatmap_overlay, input_img.size());

  addWeighted(input_img, 0.5, result_heatmap_overlay, 0.5, 0, result);

  // display_image(result, "output");

  cvtColor(result_heatmap_overlay, result_heatmap_overlay, 10);
  display_image(result_heatmap_overlay, "");
  return result_heatmap_overlay;
}

/**
 * @brief forwarding the netfile
 *
 * @param input_img : input image
 * @param output_dnn : output matrix after forwarding net
 * @param result_img_skelton : output skelton
 * 
 */
void draw_skeleton(Mat &input_img, Mat &output_dnn, Mat &result_img_skelton) {
  map<string, int> bodyparts;
  vector<pair<int, int>> pose_pairs;
  vector<Point> keypoints;

  get_pose_info(bodyparts, pose_pairs);

  for (int i = 0; i < 19; i++) {
    // A special bodypart = a special selectmap

    Mat bodypart_map =
        draw_heatmap(input_img, output_dnn, i, result_img_skelton);
    // resize(bodypart_map, bodypart_map, input_img.size());
    Point min, max;
    minMaxLoc(bodypart_map, NULL, NULL, &min, &max);
    //cout << min.x << min.y << " " << max.x << max.y << endl;
    // cout << " [" << maxInd[0] << " " << maxInd[1] << " " << endl;
    keypoints.push_back(Point(max.x, max.y));
  }

  for (int i = 0; i < 19; i++) {
    circle(result_img_skelton, keypoints[i], 10, Scalar(255, 255, 255), 2, 8,
           0);
  }

  for (const pair<int, int> &pairs : pose_pairs) {
    Point &i = keypoints[pairs.first];
    Point &j = keypoints[pairs.second];
    line(result_img_skelton, i, j, Scalar(0, 255, 0), 4);
  }
  display_image(result_img_skelton, "yooo");
}

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << " Usage: ./human_pose_estimation_video [required] openpose_graph "
            "[optional] input_video \n";
    return -1;
  }
  // decalare required input variables
  Net hpe_net;
  int input_width = 368;
  int input_height = 368;
  Mat output_dnn, result_img, input_img;
  int select_map = 0;
  VideoCapture cap;

  // [TODO] write a function to load the trained model
  load_trained_model(argv[1], hpe_net);

  // load the input video/webcam
  if (argc >= 2) {
    if (is_file_exist(argv[2])) {
      cap.open(argv[2]);
    } else {
      cap.open(0);
    }
    if (!cap.isOpened()) {
      cerr << "Unable to connect to camera" << endl;
      return 1;
    }
  }

  while (1) {
    cap >> input_img;

    if (!input_img.data) {
      cout << "Failed to acquire image \n";
      exit(0);
    }
    double t = (double)getTickCount();
    // [TODO] write a function for the forward pass
    dnn_forward_inference(input_img, hpe_net, output_dnn, input_width,
                          input_height);

    // render either keypoint or the skeleton
    if (select_map < 19) {
      // [TODO] write a function to draw the heatmap of a part
      draw_heatmap(input_img, output_dnn, select_map, result_img);
    } else {
      // [TODO] write a function to draw the complete skelton
      draw_skeleton(input_img, output_dnn, result_img);
    }

    // printing fps on the result image
    t = ((double)getTickCount() - t) / getTickFrequency();
    putText(result_img, format("speed = %.2f fps", 1.0 / t), Point(15, 15),
            FONT_HERSHEY_COMPLEX, .5, Scalar(0, 0, 0), 2);

    // display_image(input_img, "img:in");
    display_image(result_img, "img:out");
    char k = waitKey(5);
    if (k == 'q') {
      break;
    } else {
      select_map = keyboard_mapping(k, select_map);
    }
  }

  return 0;
}