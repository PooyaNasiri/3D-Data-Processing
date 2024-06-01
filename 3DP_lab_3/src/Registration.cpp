#include "Registration.h"

struct PointDistance
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // This class should include an auto-differentiable cost function.
  // To rotate a point given an axis-angle rotation, use
  // the Ceres function:
  // AngleAxisRotatePoint(...) (see ceres/rotation.h)
  // Similarly to the Bundle Adjustment case initialize the struct variables with the source and  the target point.
  // You have to optimize only the 6-dimensional array (rx, ry, rz, tx ,ty, tz).
  // WARNING: When dealing with the AutoDiffCostFunction template parameters,
  // pay attention to the order of the template parameters
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  PointDistance(const Eigen::Vector3d &source_point, const Eigen::Vector3d &target_point)
      : source_point_(source_point), target_point_(target_point) {}

  template <typename T>
  bool operator()(const T *const transform, T *residuals) const
  {
    // transform contains: [rx, ry, rz, tx, ty, tz]
    const T *rotation = transform;
    const T *translation = transform + 3;

    // Source point in T
    T source_point[3];
    source_point[0] = T(source_point_[0]);
    source_point[1] = T(source_point_[1]);
    source_point[2] = T(source_point_[2]);

    // Rotate the source point
    T rotated_point[3];
    ceres::AngleAxisRotatePoint(rotation, source_point, rotated_point);

    // Translate the point
    rotated_point[0] += translation[0];
    rotated_point[1] += translation[1];
    rotated_point[2] += translation[2];

    // Compute the residuals (difference to target point)
    residuals[0] = rotated_point[0] - T(target_point_[0]);
    residuals[1] = rotated_point[1] - T(target_point_[1]);
    residuals[2] = rotated_point[2] - T(target_point_[2]);

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d &source_point, const Eigen::Vector3d &target_point)
  {
    return (new ceres::AutoDiffCostFunction<PointDistance, 3, 6>(
        new PointDistance(source_point, target_point)));
  }

  Eigen::Vector3d source_point_;
  Eigen::Vector3d target_point_;
};

Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  open3d::io::ReadPointCloud(cloud_source_filename, source_);
  open3d::io::ReadPointCloud(cloud_target_filename, target_);
  Eigen::Vector3d gray_color;
  source_for_icp_ = source_;
}

Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  source_ = cloud_source;
  target_ = cloud_target;
  source_for_icp_ = source_;
}

void Registration::draw_registration_result()
{
  // clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  // different color
  Eigen::Vector3d color_s;
  Eigen::Vector3d color_t;
  color_s << 1, 0.706, 0;
  color_t << 0, 0.651, 0.929;

  target_clone.PaintUniformColor(color_t);
  source_clone.PaintUniformColor(color_s);
  source_clone.Transform(transformation_);

  auto src_pointer = std::make_shared<open3d::geometry::PointCloud>(source_clone);
  auto target_pointer = std::make_shared<open3d::geometry::PointCloud>(target_clone);
  open3d::visualization::DrawGeometries({src_pointer, target_pointer});
  return;
}

void Registration::execute_icp_registration(double threshold, int max_iteration, double relative_rmse, std::string mode)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // ICP main loop
  // Check convergence criteria and the current iteration.
  // If mode=="svd" use get_svd_icp_transformation if mode=="lm" use get_lm_icp_transformation.
  // Remember to update transformation_ class variable, you can use source_for_icp_ to store transformed 3d points.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int iteration = 0;
  double prev_rmse = std::numeric_limits<double>::max();
  double current_rmse = compute_rmse();

  while (iteration < max_iteration)
  {
    std::cout << "Iteration " << iteration << ", RMSE: " << current_rmse << std::endl;

    auto [source_indices, target_indices, rmse] = find_closest_point(threshold);

    Eigen::Matrix4d new_transformation = Eigen::Matrix4d::Identity();
    if (mode == "svd")
    {
      new_transformation = get_svd_icp_transformation(source_indices, target_indices);
    }
    else if (mode == "lm")
    {
      new_transformation = get_lm_icp_registration(source_indices, target_indices);
    }
    else
    {
      std::cerr << "Unknown mode: " << mode << std::endl;
      return;
    }

    transformation_ = new_transformation * transformation_;
    source_for_icp_.Transform(new_transformation);

    prev_rmse = current_rmse;
    current_rmse = compute_rmse();

    if (std::abs(prev_rmse - current_rmse) < relative_rmse)
    {
      std::cout << "Converged after " << iteration << " iterations." << std::endl;
      break;
    }

    iteration++;
  }

  if (iteration == max_iteration)
  {
    std::cout << "Reached maximum iterations without convergence." << std::endl;
  }
  return;
}

std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Find source and target indices: for each source point find the closest one in the target and discard if their
  // distance is bigger than threshold
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::vector<size_t> target_indices;
  std::vector<size_t> source_indices;
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  double total_squared_error = 0.0;
  for (size_t i = 0; i < source_for_icp_.points_.size(); ++i)
  {
    const Eigen::Vector3d &source_point = source_for_icp_.points_[i];
    std::vector<int> idx(1);
    std::vector<double> dist2(1);
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);

    if (dist2[0] <= threshold * threshold)
    {
      source_indices.push_back(i);
      target_indices.push_back(idx[0]);
      total_squared_error += dist2[0];
    }
  }

  double rmse = std::sqrt(total_squared_error / source_indices.size());
  return {source_indices, target_indices, rmse};
}

Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Find point clouds centroids and subtract them.
  // Use SVD (Eigen::JacobiSVD<Eigen::MatrixXd>) to find best rotation and translation matrix.
  // Use source_indices and target_indices to extract point to compute the 3x3 matrix to be decomposed.
  // Remember to manage the special reflection case.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Extract the corresponding points from source and target
  size_t num_correspondences = source_indices.size();
  Eigen::MatrixXd source_points(3, num_correspondences);
  Eigen::MatrixXd target_points(3, num_correspondences);

  for (size_t i = 0; i < num_correspondences; ++i)
  {
    source_points.col(i) = source_for_icp_.points_[source_indices[i]];
    target_points.col(i) = target_.points_[target_indices[i]];
  }

  // Compute centroids
  Eigen::Vector3d source_centroid = source_points.rowwise().mean();
  Eigen::Vector3d target_centroid = target_points.rowwise().mean();

  // Subtract centroids
  source_points.colwise() -= source_centroid;
  target_points.colwise() -= target_centroid;

  // Compute the covariance matrix
  Eigen::Matrix3d H = source_points * target_points.transpose();

  // Perform SVD
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  // Compute rotation matrix
  Eigen::Matrix3d R = V * U.transpose();

  // Manage the special reflection case
  if (R.determinant() < 0)
  {
    V.col(2) *= -1;
    R = V * U.transpose();
  }

  // Compute translation
  Eigen::Vector3d t = target_centroid - R * source_centroid;

  // Form the transformation matrix
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(); //(4, 4);
  transformation.block<3, 3>(0, 0) = R;
  transformation.block<3, 1>(0, 3) = t;

  return transformation;
}

Eigen::Matrix4d Registration::get_lm_icp_registration(std::vector<size_t> source_indices, std::vector<size_t> target_indices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Use LM (Ceres) to find best rotation and translation matrix.
  // Remember to convert the euler angles in a rotation matrix, store it coupled with the final translation on:
  // Eigen::Matrix4d transformation.
  // The first three elements of std::vector<double> transformation_arr represent the euler angles, the last ones
  // the translation.
  // use source_indices and target_indices to extract point to compute the matrix to be decomposed.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::vector<double> transformation_arr(6, 0.0); // 3 for rotation (Euler angles), 3 for translation

  // Initialize Ceres problem
  ceres::Problem problem;
  int num_points = source_indices.size();
  for (int i = 0; i < num_points; i++)
  {
    Eigen::Vector3d source_point = source_for_icp_.points_[source_indices[i]];
    Eigen::Vector3d target_point = target_.points_[target_indices[i]];
    ceres::CostFunction *cost_function = PointDistance::Create(source_point, target_point);
    problem.AddResidualBlock(cost_function, nullptr, transformation_arr.data());
  }

  // Configure solver options
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.max_num_iterations = 100;

  // Solve the problem
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Extract the optimized transformation
  Eigen::Matrix3d R;
  ceres::AngleAxisToRotationMatrix(transformation_arr.data(), R.data());

  Eigen::Vector3d t(transformation_arr[3], transformation_arr[4], transformation_arr[5]);

  // Form the transformation matrix
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
  transformation.block<3, 3>(0, 0) = R;
  transformation.block<3, 1>(0, 3) = t;

  return transformation;
}

void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
  transformation_ = init_transformation;
}

Eigen::Matrix4d Registration::get_transformation()
{
  return transformation_;
}

double Registration::compute_rmse()
{
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);
  int num_source_points = source_clone.points_.size();
  Eigen::Vector3d source_point;
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  double mse;
  for (size_t i = 0; i < num_source_points; ++i)
  {
    source_point = source_clone.points_[i];
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
    mse = mse * i / (i + 1) + dist2[0] / (i + 1);
  }
  return sqrt(mse);
}

void Registration::write_tranformation_matrix(std::string filename)
{
  std::ofstream outfile(filename);
  if (outfile.is_open())
  {
    outfile << transformation_;
    outfile.close();
  }
}

void Registration::save_merged_cloud(std::string filename)
{
  // clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  source_clone.Transform(transformation_);
  open3d::geometry::PointCloud merged = target_clone + source_clone;
  open3d::io::WritePointCloud(filename, merged);
}
