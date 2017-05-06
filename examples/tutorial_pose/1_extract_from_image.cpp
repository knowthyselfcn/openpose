// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

#include <std_srvs/Empty.h>
#include <ros/node_handle.h>
#include <ros/service_server.h>
#include <ros/init.h>
#include <cv_bridge/cv_bridge.h>

#include <image_recognition_msgs/GetPersons.h>

std::shared_ptr<op::PoseExtractor> g_pose_extractor;
std::map<unsigned char, std::string> g_bodypart_map;
std::string g_model_folder;
cv::Size g_net_input_size;
cv::Size g_net_output_size;
cv::Size g_output_size;
unsigned int g_num_scales;
unsigned int g_num_gpu_start;
double g_scale_gap;
double g_alpha_pose;
op::PoseModel g_pose_model;

//!
//! \brief getParam Get parameter from node handle
//! \param nh The nodehandle
//! \param param_name Key string
//! \param default_value Default value if not found
//! \return The parameter value
//!
template <typename T>
T getParam(const ros::NodeHandle& nh, const std::string& param_name, T default_value)
{
  T value;
  if (nh.hasParam(param_name))
  {
    nh.getParam(param_name, value);
  }
  else
  {
    ROS_WARN_STREAM("Parameter '" << param_name << "' not found, defaults to '" << default_value << "'");
    value = default_value;
  }
  return value;
}

op::PoseModel stringToPoseModel(const std::string& pose_model_string)
{
  if (pose_model_string == "COCO")
    return op::PoseModel::COCO_18;
  else if (pose_model_string == "MPI")
    return op::PoseModel::MPI_15;
  else if (pose_model_string == "MPI_4_layers")
    return op::PoseModel::MPI_15_4;
  else
  {
    ROS_ERROR("String does not correspond to any model (COCO, MPI, MPI_4_layers)");
    return op::PoseModel::COCO_18;
  }
}

std::map<unsigned char, std::string> getBodyPartMapFromPoseModel(const op::PoseModel& pose_model)
{
  if (pose_model == op::PoseModel::COCO_18)
  {
    return op::POSE_COCO_BODY_PARTS;
  }
  else if (pose_model == op::PoseModel::MPI_15 || pose_model == op::PoseModel::MPI_15_4)
  {
    return op::POSE_MPI_BODY_PARTS;
  }
  else
  {
    ROS_FATAL("Invalid pose model, not map present");
    exit(1);
  }
}

image_recognition_msgs::BodypartDetection getBodyPartDetectionFromArrayAndIndex(const op::Array<float>& array, size_t idx)
{
  image_recognition_msgs::BodypartDetection bodypart;
  bodypart.x = array[idx];
  bodypart.y = array[idx+1];
  bodypart.confidence = array[idx+2];
  return bodypart;
}

image_recognition_msgs::BodypartDetection getNANBodypart()
{
  image_recognition_msgs::BodypartDetection bodypart;
  bodypart.confidence = NAN;
  return bodypart;
}

bool detectPosesCallback(image_recognition_msgs::GetPersons::Request& req, image_recognition_msgs::GetPersons::Response& res)
{
    ROS_INFO("detectPosesCallback");

    // Convert ROS message to opencv image
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(req.image, req.image.encoding);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("detectPosesCallback cv_bridge exception: %s", e.what());
        return false;
    }
    cv::Mat image = cv_ptr->image;
    if(image.empty())
    {
      ROS_ERROR("Empty image!");
      return false;
    }

    // Step 3 - Initialize all required classes
    op::CvMatToOpInput cv_mat_to_input{g_net_input_size, g_num_scales, (float) g_scale_gap};
    op::CvMatToOpOutput cv_mat_to_output{g_output_size};

    op::PoseRenderer pose_renderer{g_net_output_size, g_output_size, g_pose_model, nullptr, (float) g_alpha_pose};
    op::OpOutputToCvMat op_output_to_cv_mat{g_output_size};
    const cv::Size windowed_size = g_output_size;
    //op::FrameDisplayer frame_displayer{windowed_size, "OpenPose Tutorial - Example 1"};

    pose_renderer.initializationOnThread();


    // Step 2 - Format input image to OpenPose input and output formats
    const auto net_input_array = cv_mat_to_input.format(image);
    double scale_input_to_output;
    op::Array<float> output_array;
    std::tie(scale_input_to_output, output_array) = cv_mat_to_output.format(image);
    // Step 3 - Estimate poseKeyPoints
    g_pose_extractor->forwardPass(net_input_array, image.size());
    const auto pose_keypoints = g_pose_extractor->getPoseKeyPoints();
    // Step 4 - Render poseKeyPoints
    pose_renderer.renderPose(output_array, pose_keypoints);
    // Step 5 - OpenPose output format to cv::Mat
    auto output_image = op_output_to_cv_mat.formatToCvMat(output_array);

    // ------------------------- SHOWING RESULT AND CLOSING -------------------------
    // Step 1 - Show results
    //frame_displayer.displayFrame(output_image, 0); // Alternative: cv::imshow(outputImage) + cv::waitKey(0)

    if (!pose_keypoints.empty() && pose_keypoints.getNumberDimensions() != 3)
  {
    ROS_ERROR("pose.getNumberDimensions(): %d != 3", (int) pose_keypoints.getNumberDimensions());
    return false;
  }

  int num_people = pose_keypoints.getSize(0);
  int num_bodyparts = pose_keypoints.getSize(1);

  for (size_t person_idx = 0; person_idx < num_people; person_idx++)
  {
    // Initialize all bodyparts with nan
    image_recognition_msgs::PersonDetection person_msg;
    person_msg.nose = getNANBodypart();
    person_msg.neck = getNANBodypart();
    person_msg.right_shoulder = getNANBodypart();
    person_msg.right_elbow = getNANBodypart();
    person_msg.right_wrist = getNANBodypart();
    person_msg.left_shoulder = getNANBodypart();
    person_msg.left_elbow = getNANBodypart();
    person_msg.left_wrist = getNANBodypart();
    person_msg.right_hip = getNANBodypart();
    person_msg.right_knee = getNANBodypart();
    person_msg.right_ankle = getNANBodypart();
    person_msg.left_hip = getNANBodypart();
    person_msg.left_knee = getNANBodypart();
    person_msg.left_ankle = getNANBodypart();
    person_msg.right_eye = getNANBodypart();
    person_msg.left_eye = getNANBodypart();
    person_msg.right_ear = getNANBodypart();
    person_msg.left_ear = getNANBodypart();
    person_msg.chest = getNANBodypart();

    for (size_t bodypart_idx = 0; bodypart_idx < num_bodyparts; bodypart_idx++)
    {
      size_t final_idx = 3*(person_idx*num_bodyparts + bodypart_idx);
      image_recognition_msgs::BodypartDetection bodypart_detection = getBodyPartDetectionFromArrayAndIndex(pose_keypoints, final_idx);

      std::string body_part_string = g_bodypart_map[bodypart_idx];

      if (body_part_string == "Nose") person_msg.nose = bodypart_detection;
      else if (body_part_string == "Neck") person_msg.neck = bodypart_detection;
      else if (body_part_string == "RShoulder") person_msg.right_shoulder = bodypart_detection;
      else if (body_part_string == "RElbow") person_msg.right_elbow = bodypart_detection;
      else if (body_part_string == "RWrist") person_msg.right_wrist = bodypart_detection;
      else if (body_part_string == "LShoulder") person_msg.left_shoulder = bodypart_detection;
      else if (body_part_string == "LElbow") person_msg.left_elbow = bodypart_detection;
      else if (body_part_string == "LWrist") person_msg.left_wrist = bodypart_detection;
      else if (body_part_string == "RHip") person_msg.right_hip = bodypart_detection;
      else if (body_part_string == "RKnee") person_msg.right_knee = bodypart_detection;
      else if (body_part_string == "RAnkle") person_msg.right_ankle = bodypart_detection;
      else if (body_part_string == "LHip") person_msg.left_hip = bodypart_detection;
      else if (body_part_string == "LKnee") person_msg.left_knee = bodypart_detection;
      else if (body_part_string == "LAnkle") person_msg.left_ankle = bodypart_detection;
      else if (body_part_string == "REye") person_msg.right_eye = bodypart_detection;
      else if (body_part_string == "LEye") person_msg.left_eye = bodypart_detection;
      else if (body_part_string == "REar") person_msg.right_ear = bodypart_detection;
      else if (body_part_string == "LEar") person_msg.left_ear = bodypart_detection;
      else if (body_part_string == "Chest") person_msg.chest = bodypart_detection;
      else
      {
        ROS_ERROR("Unknown bodypart %s, this should never happen!", body_part_string.c_str());
      }
    }
    res.detections.push_back(person_msg);
  }

  ROS_INFO("Detected %d persons", (int) res.detections.size());

  return true;
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "image_recognition_msgs");

    ros::NodeHandle local_nh("~");

    g_net_input_size = cv::Size(getParam(local_nh, "net_input_width", 656), getParam(local_nh, "net_input_height", 368));
    g_net_output_size = cv::Size(getParam(local_nh, "net_output_width", 656), getParam(local_nh, "net_output_height", 368));
    g_output_size = cv::Size(getParam(local_nh, "output_width", 1280), getParam(local_nh, "output_height", 720));
    g_num_scales = getParam(local_nh, "num_scales", 1);
    g_scale_gap = getParam(local_nh, "scale_gap", 0.3);
    g_num_gpu_start = getParam(local_nh, "num_gpu_start", 0);
    g_model_folder = getParam(local_nh, "model_folder", std::string("/home/rein/openpose/models/"));
    g_pose_model = stringToPoseModel(getParam(local_nh, "pose_model", std::string("COCO")));
    g_bodypart_map = getBodyPartMapFromPoseModel(g_pose_model);
    g_alpha_pose = getParam(local_nh, "alpha_pose", 0.6);

    ros::NodeHandle nh;
    ros::ServiceServer service = nh.advertiseService("detect_poses", detectPosesCallback);

    g_pose_extractor = std::shared_ptr<op::PoseExtractorCaffe>(
        new op::PoseExtractorCaffe(g_net_input_size, g_net_output_size, g_output_size, g_num_scales, (float) g_scale_gap, g_pose_model,
                                              g_model_folder, g_num_gpu_start));
    g_pose_extractor->initializationOnThread();
    

    ros::spin();

    return 0;
}
