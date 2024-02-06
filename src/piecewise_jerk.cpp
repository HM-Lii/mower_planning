#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>

#include "OsqpEigen/OsqpEigen.h"
using namespace std;
using namespace Eigen;
struct Interval {
  double lower, upper;
  Interval(double l, double u) : lower(l), upper(u) {}
  double width() const { return upper - lower; }
};
class PathPlanner {
 public:
  PathPlanner(ros::NodeHandle& nh) {
    init();
    // map_sub_ = nh.subscribe("map", 1, &PathPlanner::mapCallback, this);
    path_pub_ = nh.advertise<nav_msgs::Path>("path", 1);
    bound_pub_ = nh.advertise<visualization_msgs::MarkerArray>("bound", 1);
    
    // 提取边界
    time_start = ros::Time::now();
    BoundDecider();
    std::cout << "bound time use:" << (ros::Time::now() - time_start).toSec()
              << "s" << std::endl;
    // 构建QP问题
    QpProblem();
    visualize();
    ros::spin();
  }
  ~PathPlanner(){};

 private:
  int n, l_ref_bound;
  double s_len, delta_s, l_bound, dddl_bound, w_l, w_dl, w_ddl, w_dddl, l_init,
      dl_init;
  Eigen::VectorXd x, up_bound, low_bound, s_ref, QPSolution;
  ros::NodeHandle nh_;
  ros::Subscriber map_sub_;
  ros::Publisher path_pub_;
  ros::Publisher bound_pub_;
  ros::Time time_start, time_end;
  nav_msgs::OccupancyGrid map_;
  void init() {
    l_init = -1;
    dl_init = 0;
    s_len = 140;
    delta_s = 1;
    n = int(s_len / delta_s);
    x = Eigen::VectorXd::LinSpaced(n, -140, s_len);

    l_bound = 25;
    l_ref_bound = 20;
    dddl_bound = 0.01;
    up_bound = Eigen::VectorXd::Zero(5 * n + 3);
    low_bound = Eigen::VectorXd::Zero(5 * n + 3);
    s_ref = Eigen::VectorXd::Zero(3 * n);
    w_l = 0.01;
    w_dl = 0.1;
    w_ddl = 4;
    w_dddl = 0.1;
    low_bound[5 * n] = 1;
    up_bound[5 * n] = 1;
  }
  void QpProblem() {
    // 构造P和Q
    MatrixXd eye_n = MatrixXd::Identity(n, n);
    MatrixXd zero_n = MatrixXd::Zero(n, n);
    MatrixXd tri_n = MatrixXd::Zero(n, n);
    tri_n.diagonal(1).array() = 1;

    MatrixXd P_zeros = zero_n;
    MatrixXd P_l = w_l * eye_n;
    MatrixXd P_dl = w_dl * eye_n;
    MatrixXd P_ddl = (w_ddl + 2 * w_dddl / (delta_s * delta_s)) * eye_n -
                     2 * w_dddl / (delta_s * delta_s) * tri_n;
    P_ddl(n - 1, n - 1) = w_ddl + w_dddl / (delta_s * delta_s);

    MatrixXd P(3 * n, 3 * n);
    P << P_l, P_zeros, P_zeros, P_zeros, P_dl, P_zeros, P_zeros, P_zeros, P_ddl;

    VectorXd q = -w_l * s_ref;
    // 构造：l(i+1) = l(i) + l'(i) * delta_s + 1/2 * l''(i) * delta_s^2 + 1/6 *
    // l'''(i) * delta_s^3
    MatrixXd A_ll = -eye_n + tri_n;
    MatrixXd A_ldl = -delta_s * eye_n;
    MatrixXd A_lddl = -0.5 * delta_s * delta_s * eye_n;
    MatrixXd A_l = MatrixXd::Zero(n, 3 * n);
    A_l.block(0, 0, n, n) = A_ll;
    A_l.block(0, n, n, n) = A_ldl;
    A_l.block(0, 2 * n, n, n) = A_lddl;

    // 构造：l'(i+1) = l'(i) + l''(i) * delta_s + 1/2 * l'''(i) * delta_s^2
    MatrixXd A_dll = zero_n;
    MatrixXd A_dldl = -eye_n + tri_n;
    MatrixXd A_dlddl = -delta_s * eye_n;
    MatrixXd A_dl = MatrixXd::Zero(n, 3 * n);
    A_dl.block(0, 0, n, n) = A_dll;
    A_dl.block(0, n, n, n) = A_dldl;
    A_dl.block(0, 2 * n, n, n) = A_dlddl;

    MatrixXd A_ul = MatrixXd::Zero(3 * n, 3 * n);
    A_ul.block(0, 0, n, n) = eye_n;

    // 初始化设置
    MatrixXd A_init = MatrixXd::Zero(3, 3 * n);
    A_init(0, 0) = l_init;
    A_init(0, 1) = dl_init;

    MatrixXd A = MatrixXd::Zero(5 * n + 3, 3 * n);
    A.block(0, 0, 3 * n, 3 * n) = A_ul;
    A.block(3 * n, 0, n, 3 * n) = A_l;
    A.block(4 * n, 0, n, 3 * n) = A_dl;
    A.block(5 * n, 0, 3, 3 * n) = A_init;

    VectorXd l = low_bound;
    VectorXd u = up_bound;
    SparseMatrix<double> P_sparse = P.sparseView();
    SparseMatrix<double> A_sparse = A.sparseView();
    // 创建OSQP对象并求解问题
    osqpSolver(n, P_sparse, q, A_sparse, l, u);
  }
  void osqpSolver(int n, SparseMatrix<double> P_sparse, VectorXd q,
                  SparseMatrix<double> A_sparse, VectorXd l, VectorXd u) {
    // 创建求解器实例
    OsqpEigen::Solver solver;

    // 设置参数
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);

    // 设置QP求解器的初始数据
    solver.data()->setNumberOfVariables(3 * n);
    solver.data()->setNumberOfConstraints(5 * n + 3);
    if (!solver.data()->setHessianMatrix(P_sparse)) return;
    if (!solver.data()->setGradient(q)) return;
    if (!solver.data()->setLinearConstraintsMatrix(A_sparse)) return;
    if (!solver.data()->setLowerBound(l)) return;
    if (!solver.data()->setUpperBound(u)) return;

    // 初始化求解器
    if (!solver.initSolver()) return;

    time_start = ros::Time::now();
    // 求解QP问题
    if (!solver.solve()) return;
    time_end = ros::Time::now();
    // 获取解
    QPSolution = solver.getSolution();
    // std::cout << "QPSolution" << std::endl << QPSolution << std::endl;
    std::cout << "osqp time use:" << (time_end - time_start).toSec() << "s"
              << std::endl;
  }
  void mapCallback(const nav_msgs::OccupancyGrid& msg) {
    time_start = ros::Time::now();

    BoundDecider(msg);
    std::cout << "bound time use:" << (ros::Time::now() - time_start).toSec()
              << "s" << std::endl;
    // QpProblem();
    visualize();
  }
    void BoundDecider() {
      std::vector<std::vector<double>> obs = {{0, 9, 25, -25},
      {10, 22, -10, -25},{23, 39, 25, -25},{40, 52, -8, -25},    
      {53, 69, 25, -25}, 
  {70, 81, 25, -5},{81, 100, 25, -25}
};
      for (int i = 0; i < n; ++i) {
        double up_, low_;
        for (const auto& ob : obs) {
          if (x[i] >= ob[0] && x[i] <= ob[1]) {
            low_ = ob[3];
            up_ = ob[2];
            break;
          } else {
            up_ = l_bound;
            low_ = -l_bound;
          }
        }
        up_bound[i] = up_;
        low_bound[i] = low_;
        s_ref[i] = 0.5 * (up_ + low_);
      }
  }
  void BoundDecider(const nav_msgs::OccupancyGrid& map_) {
    double s_ref_, up_, low_;
    vector<map<double, pair<int, int>>> sec;
    vector<pair<int, int>> bound(n);

    std::vector<Interval> safe_intervals;
    double y_center = 0.0;  // 初始y坐标

    for (int i = 0; i < n; ++i) {
      std::vector<Interval> feasible_intervals;
      feasible_intervals.emplace_back(
          std::max(y_center - l_ref_bound, -l_bound),
          std::min(y_center + l_ref_bound, l_bound));

      for (int row_height = -l_bound; row_height <= l_bound; ++row_height) {
        int index = (row_height + (map_.info.height / 2)) * map_.info.width +
                    i+map_.info.width / 2;
        if (map_.data[index] == 100) {
          cout << "index:" << index << "; row_height:" << row_height
               << "; col: " << i << endl;
        }

        for (auto it = feasible_intervals.begin();
             it != feasible_intervals.end();) {
              cout<<"feasible_intervals:"<<it->lower<<","<<it->upper<<endl;
          if (map_.data[index] == 100 && row_height > it->lower &&
              row_height < it->upper) {
            Interval upper_interval(row_height, it->upper);
            it->upper = row_height;
            cout<<"upper_interval:"<<upper_interval.lower<<","<<upper_interval.upper<<endl;
            if (upper_interval.width() > l_ref_bound / 4)
              feasible_intervals.insert(it, upper_interval);
          }
          if (it->width() < l_ref_bound / 4)
            it = feasible_intervals.erase(it);
          else
            ++it;
        }
      }
      cout<<"observable intervals:"<<feasible_intervals.size()<<endl;
      if (feasible_intervals.empty()) {
        // 回溯
        while (!safe_intervals.empty()) {
          Interval last_interval = safe_intervals.back();
          safe_intervals.pop_back();
          if (last_interval.width() > l_ref_bound / 2) {
            safe_intervals.emplace_back(
                last_interval.lower, last_interval.upper);
            y_center = (last_interval.lower + last_interval.upper) /
                       2;
            break;
          }
        }
        if (safe_intervals.empty()) {
          cout << "Search failed!" << endl;
          // 搜索失败
          return ;
        }
      } else {
        // 选择最宽的区间
        Interval max_interval = *std::max_element(
            feasible_intervals.begin(), feasible_intervals.end(),
            [](Interval a,Interval b){ return a.width() < b.width(); });
        safe_intervals.push_back(max_interval);
        y_center = (max_interval.lower + max_interval.upper) / 2;
      }
    }
    for (int i = 0; i < n; ++i) {
      up_bound[i] = safe_intervals[i].upper;
      low_bound[i] = safe_intervals[i].lower;
      s_ref[i] = 0.5 * (up_bound[i] + low_bound[i]);
    }
  }
  void visualize() {
    visualization_msgs::MarkerArray marker_array;
    for (int i = 0; i < x.size(); ++i) {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "camera_init";
      marker.header.stamp = ros::Time::now();
      marker.ns = "up_bounds";
      marker.id = i;
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x = 0.1 * x[i];
      marker.pose.position.y = 0.1* up_bound[i];
      marker.pose.position.z = 0;
      marker.scale.x = 0.1;
      marker.scale.y = 0.1;
      marker.scale.z = 0.1;
      marker.color.a = 1.0;
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;

      marker_array.markers.push_back(marker);
      marker.ns = "low_bounds";
      marker.id = i;
      marker.pose.position.y = 0.1 * low_bound[i];
      marker.color.r = 1.0;
      marker.color.g = 1.0;
      marker_array.markers.push_back(marker);

      marker_array.markers.push_back(marker);
      marker.ns = "path";
      marker.id = i;
      marker.pose.position.y = 0.1 * QPSolution[i];
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker_array.markers.push_back(marker);
    }
    ros::Rate loop_rate(1);
    while (ros::ok()) {
      bound_pub_.publish(marker_array);
      loop_rate.sleep();
      ros::spinOnce();
    }
  }
  };
int main(int argc, char** argv) {
  ros::init(argc, argv, "path_planning");
  ros::NodeHandle nh;
  PathPlanner path_planner(nh);
  return 0;
}
