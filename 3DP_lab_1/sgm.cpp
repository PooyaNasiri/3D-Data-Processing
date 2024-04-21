#include <algorithm>
#include <vector>
#include <cmath>
#include <ctime>
#include <thread>
#include <chrono>
#include <algorithm>

#include "sgm.h"
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#define NUM_DIRS 3
#define PATHS_PER_SCAN 8

using namespace std;
using namespace cv;
using namespace Eigen;
static char hamLut[256][256];
static int directions[NUM_DIRS] = {0, -1, 1};

// compute values for hamming lookup table
void compute_hamming_lut()
{
  for (uchar i = 0; i < 255; i++)
  {
    for (uchar j = 0; j < 255; j++)
    {
      uchar census_xor = i ^ j;
      uchar dist = 0;
      while (census_xor)
      {
        ++dist;
        census_xor &= census_xor - 1;
      }

      hamLut[i][j] = dist;
    }
  }
}

namespace sgm
{
  SGM::SGM(unsigned int disparity_range, unsigned int p1, unsigned int p2, float conf_thresh, unsigned int window_height, unsigned window_width) : disparity_range_(disparity_range), p1_(p1), p2_(p2), conf_thresh_(conf_thresh), window_height_(window_height), window_width_(window_width)
  {
    compute_hamming_lut();
  }

  // set images and initialize all the desired values
  void SGM::set(const cv::Mat &left_img, const cv::Mat &right_img, const cv::Mat &right_mono)
  {
    views_[0] = left_img;
    views_[1] = right_img;
    mono_ = right_mono;

    height_ = left_img.rows;
    width_ = right_img.cols;
    pw_.north = window_height_ / 2;
    pw_.south = height_ - window_height_ / 2;
    pw_.west = window_width_ / 2;
    pw_.east = width_ - window_width_ / 2;
    init_paths();
    cost_.resize(height_, ul_array2D(width_, ul_array(disparity_range_)));
    inv_confidence_.resize(height_, vector<float>(width_));
    aggr_cost_.resize(height_, ul_array2D(width_, ul_array(disparity_range_)));
    path_cost_.resize(PATHS_PER_SCAN, ul_array3D(height_, ul_array2D(width_, ul_array(disparity_range_))));
  }

  // initialize path directions
  void SGM::init_paths()
  {
    for (int i = 0; i < NUM_DIRS; ++i)
      for (int j = 0; j < NUM_DIRS; ++j)
      {
        // skip degenerate path
        if (i == 0 && j == 0)
          continue;
        paths_.push_back({directions[i], directions[j]});
      }
  }

  // compute costs and fill volume cost cost_
  void SGM::calculate_cost_hamming()
  {
    cv::Mat_<uchar> census_img[2];
    cv::Mat_<uchar> census_mono[2];
    cout << "\nApplying Census Transform" << endl;

    for (int view = 0; view < 2; view++)
    {
      census_img[view] = cv::Mat_<uchar>::zeros(height_, width_);
      census_mono[view] = cv::Mat_<uchar>::zeros(height_, width_);
      for (int r = 1; r < height_ - 1; r++)
      {
        uchar *p_center = views_[view].ptr<uchar>(r),
              *p_census = census_img[view].ptr<uchar>(r);
        p_center += 1;
        p_census += 1;
        for (int c = 1; c < width_ - 1; c++, p_center++, p_census++)
        {
          uchar p_census_val = 0, m_census_val = 0, shift_count = 0;
          for (int wr = r - 1; wr <= r + 1; wr++)
          {
            for (int wc = c - 1; wc <= c + 1; wc++)
            {

              if (shift_count != 4) // skip the center pixel
              {
                p_census_val <<= 1;
                m_census_val <<= 1;
                if (views_[view].at<uchar>(wr, wc) < *p_center) // compare pixel values in the neighborhood
                  p_census_val = p_census_val | 0x1;
              }
              shift_count++;
            }
          }
          *p_census = p_census_val;
        }
      }
    }

    cout << "\nFinding Hamming Distance" << endl;

    for (unsigned int r = window_height_ / 2 + 1; r < height_ - window_height_ / 2 - 1; r++)
      for (unsigned int c = window_width_ / 2 + 1; c < width_ - window_width_ / 2 - 1; c++)
        for (unsigned int d = 0; d < disparity_range_; d++)
        {
          long cost = 0;
          for (unsigned int wr = r - window_height_ / 2; wr <= r + window_height_ / 2; wr++)
          {
            uchar *p_left = census_img[0].ptr<uchar>(wr),
                  *p_right = census_img[1].ptr<uchar>(wr);
            unsigned int wc = c - window_width_ / 2;
            p_left += wc;
            p_right += wc + d;
            const uchar out_val = census_img[1].at<uchar>(wr, width_ - window_width_ / 2 - 1);
            for (; wc <= c + window_width_ / 2; wc++, p_left++, p_right++)
            {
              uchar census_left, census_right;
              census_left = *p_left;
              if (c + d < width_ - window_width_ / 2)
                census_right = *p_right;
              else
                census_right = out_val;
              cost += ((hamLut[census_left][census_right]));
            }
          }
          cost_[r][c][d] = cost;
        }
  }

  void SGM::compute_path_cost(int direction_y, int direction_x, int cur_y, int cur_x, int cur_path)
  {
    unsigned long prev_cost, best_prev_cost, penalty_cost,
        small_penalty_cost, big_penalty_cost;

    //////////////////////////// Code to be completed (1/4) /////////////////////////////////

    for (unsigned int i = 0; i < disparity_range_; i++)
      if (cur_y == pw_.north || cur_y == pw_.south || cur_x == pw_.east || cur_x == pw_.west)
        path_cost_[cur_path][cur_y][cur_x][i] = cost_[cur_y][cur_x][i];
      else
      {
        prev_cost = path_cost_[cur_path][cur_y - direction_y][cur_x - direction_x][i];
        best_prev_cost = prev_cost;
        for (unsigned int j = 0; j < disparity_range_; j++)
        {
          if (abs(i - j) == 1)
            small_penalty_cost = prev_cost + p1_;
          else if (abs(i - j) > 1)
            big_penalty_cost = prev_cost + p2_;
          else if (abs(i - j) == 0)
            penalty_cost = prev_cost;

          best_prev_cost = std::min({best_prev_cost, small_penalty_cost, big_penalty_cost, penalty_cost});
        }
        path_cost_[cur_path][cur_y][cur_x][i] = cost_[cur_y][cur_x][i] + best_prev_cost;
      }
  }

  void SGM::aggregation()
  {
    vector<pair<int, int>> all_paths = {
        {0, -1}, {0, 1}, // Left to right and right to left
        {-1, 0},
        {1, 0}, // Up to down and down to up
        {-1, -1},
        {-1, 1}, // Top-left to bottom-right and top-right to bottom-left
        {1, -1},
        {1, 1} // Bottom-left to top-right and bottom-right to top-left
    };

    for (long unsigned int cur_path = 0; cur_path < all_paths.size(); ++cur_path)
    {
      //////////////////////////// Code to be completed (2/4) /////////////////////////////////
      int dir_y = all_paths[cur_path].first;
      int dir_x = all_paths[cur_path].second;
      int start_y, start_x, end_y, end_x, step_y, step_x;

      if (dir_y == 0 && dir_x == -1)
      {
        // Left to right
        start_y = 0;
        start_x = pw_.west;
        end_y = height_;
        end_x = pw_.east;
        step_y = 1;
        step_x = 1;
      }
      else if (dir_y == 0 && dir_x == 1)
      {
        // Right to left
        start_y = 0;
        start_x = pw_.east;
        end_y = height_;
        end_x = pw_.west;
        step_y = 1;
        step_x = -1;
      }
      else if (dir_y == -1 && dir_x == 0)
      {
        // Up to down
        start_y = pw_.north;
        start_x = 0;
        end_y = pw_.south;
        end_x = width_;
        step_y = 1;
        step_x = 1;
      }
      else if (dir_y == 1 && dir_x == 0)
      {
        // Down to up
        start_y = pw_.south;
        start_x = 0;
        end_y = pw_.north;
        end_x = width_;
        step_y = -1;
        step_x = 1;
      }
      else if (dir_y == -1 && dir_x == -1)
      {
        // Top-left to bottom-right
        start_y = pw_.north;
        start_x = pw_.west;
        end_y = pw_.south;
        end_x = pw_.east;
        step_y = 1;
        step_x = 1;
      }
      else if (dir_y == -1 && dir_x == 1)
      {
        // Top-right to bottom-left
        start_y = pw_.north;
        start_x = pw_.east;
        end_y = pw_.south;
        end_x = pw_.west;
        step_y = 1;
        step_x = -1;
      }
      else if (dir_y == 1 && dir_x == -1)
      {
        // Bottom-left to top-right
        start_y = pw_.south;
        start_x = pw_.west;
        end_y = pw_.north;
        end_x = pw_.east;
        step_y = -1;
        step_x = 1;
      }
      else if (dir_y == 1 && dir_x == 1)
      {
        // Bottom-right to top-left
        start_y = pw_.south;
        start_x = pw_.east;
        end_y = pw_.north;
        end_x = pw_.west;
        step_y = -1;
        step_x = -1;
      }

      // Process pixels along the specified path
      for (int y = start_y; y != end_y; y += step_y)
        for (int x = start_x; x != end_x; x += step_x)
          compute_path_cost(dir_y, dir_x, y, x, cur_path);
    }

    float alpha = (PATHS_PER_SCAN - 1) / static_cast<float>(PATHS_PER_SCAN);
    // Final aggregation and confidence update
    for (int row = 0; row < height_; ++row)
      for (int col = 0; col < width_; ++col)
        for (int path = 0; path < PATHS_PER_SCAN; path++)
        {
          unsigned long min_on_path = path_cost_[path][row][col][0];
          int disp = 0;

          // Accumulate path costs and find minimum along each disparity
          for (unsigned int d = 0; d < disparity_range_; d++)
          {
            aggr_cost_[row][col][d] += path_cost_[path][row][col][d];
            if (path_cost_[path][row][col][d] < min_on_path)
            {
              min_on_path = path_cost_[path][row][col][d];
              disp = d;
            }
          }

          // Update inverse confidence
          inv_confidence_[row][col] += (min_on_path - alpha * cost_[row][col][disp]);
        }
  }

  void SGM::compute_disparity()
  {
    calculate_cost_hamming();
    aggregation();
    disp_ = Mat(Size(width_, height_), CV_8UC1, Scalar::all(0));
    vector<unsigned long> good_costs;
    vector<float> unscaled_disparities;
    for (int row = 0; row < height_; ++row)
    {
      for (int col = 0; col < width_; ++col)
      {
        unsigned long smallest_cost = aggr_cost_[row][col][0];
        int smallest_disparity = 0;
        for (int d = disparity_range_ - 1; d >= 0; --d)
          if (aggr_cost_[row][col][d] < smallest_cost)
          {
            smallest_cost = aggr_cost_[row][col][d];
            smallest_disparity = d;
          }
        inv_confidence_[row][col] = smallest_cost - inv_confidence_[row][col];
        if (inv_confidence_[row][col] > 0 && inv_confidence_[row][col] < conf_thresh_)
        {
          //////////////////////////// Code to be completed (3/4) /////////////////////////////////
          good_costs.push_back(smallest_disparity * 255.0 / disparity_range_);
          unscaled_disparities.push_back(static_cast<float>(mono_.at<uchar>(row, col)));
        }
        disp_.at<uchar>(row, col) = static_cast<uchar>(smallest_disparity * 255.0 / disparity_range_);
      }
    }

    //////////////////////////// Code to be completed (4/4) /////////////////////////////////
    int n = good_costs.size();
    if (n > 0)
    {
      MatrixXf A(n, 2);
      VectorXf b(n);
      for (int i = 0; i < n; ++i)
      {
        A(i, 0) = unscaled_disparities[i];
        A(i, 1) = 1;
        b(i) = good_costs[i];
      }
      VectorXf x = (A.transpose() * A).inverse() * A.transpose() * b;

      for (int row = 0; row < height_; ++row)
        for (int col = 0; col < width_; ++col)
        {
          int initial_guess_disparity = mono_.at<uchar>(row, col);
          int scaled_disparity = static_cast<int>(x(0) * initial_guess_disparity + x(1));
          if (inv_confidence_[row][col] <= 0 || inv_confidence_[row][col] >= conf_thresh_)
            disp_.at<uchar>(row, col) = scaled_disparity;
        }
    }
  }

  float SGM::compute_mse(const cv::Mat &gt)
  {
    cv::Mat1f container[2];
    cv::normalize(gt, container[0], 0, 85, cv::NORM_MINMAX);
    cv::normalize(disp_, container[1], 0, disparity_range_, cv::NORM_MINMAX);

    cv::Mat1f mask = min(gt, 1);
    cv::multiply(container[1], mask, container[1], 1);
    float error = 0;
    for (int y = 0; y < height_; ++y)
      for (int x = 0; x < width_; ++x)
      {
        float diff = container[0](y, x) - container[1](y, x);
        error += (diff * diff);
      }
    error = error / (width_ * height_);
    return error;
  }

  void SGM::save_disparity(char *out_file_name)
  {
    imwrite(out_file_name, disp_);
    return;
  }
}
