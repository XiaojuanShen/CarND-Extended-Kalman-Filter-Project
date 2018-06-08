#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_radar_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  //measurement matrix - laser
  H_laser_ << 1,0,0,0,
        0,1,0,0;

  //measurement matrix - laser
  Hj_radar_ << 1,1,0,0,
        1,1,0,0,
        1,1,0,0;

  //initializing KalmanFiler objects
  ekf_.P_ = MatrixXd(4,4);
  ekf_.F_ = MatrixXd(4,4);
  ekf_.Q_ = MatrixXd(4,4);

  //state covariance matrix
  ekf_.P_ << 1,0,0,0,
      0,1,0,0,
      0,0,1000,0,
      0,0,0,1000;

  //state transition matrix, delta_t taken 1
  ekf_.F_ << 1,0,1,0,
      0,1,0,1,
      0,0,1,0,
      0,0,0,1;

  //process covariance matrix
  ekf_.Q_ << 0,0,0,0,
      0,0,0,0,
      0,0,0,0,
      0,0,0,0;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates.
      */
      float rho = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      float rho_dot = measurement_pack.raw_measurements_(2);

      //initialize state
      ekf_.x_(0) = rho * cos(phi);
      ekf_.x_(1) = rho * sin(phi);
      ekf_.x_(2) = rho_dot * cos(phi);
      ekf_.x_(3) = rho_dot * sin(phi);

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      ekf_.x_(0) = measurement_pack.raw_measurements_(0); //px
      ekf_.x_(1) = measurement_pack.raw_measurements_(1); //py
      ekf_.x_(2) = 0; //vx
      ekf_.x_(3) = 0; //vy
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;

    // set previous_timestamp_
    previous_timestamp_ = measurement_pack.timestamp_;

    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  // compute elapsed time and convert to seconds
  float d_t = (measurement_pack.timestamp_ - previous_timestamp_)/1000000.0;

  // 1. set previous_timestamp_
  previous_timestamp_ = measurement_pack.timestamp_;

  // 2. update the state transition matrix F
  ekf_.F_(0,2) = d_t;
  ekf_.F_(1,3) = d_t;

  // 3. update the process noise covariance matrix.
  float d_t2 = pow(d_t, 2);
  float d_t3 = pow(d_t, 3);
  float d_t4 = pow(d_t, 4);

  float noise_ax = 9;
  float noise_ay = 9;

  ekf_.Q_ << d_t4 / 4 * noise_ax, 0, d_t3 / 2 * noise_ax, 0,
         0, d_t4 / 4 * noise_ay, 0, d_t3 / 2 * noise_ay,
         d_t3 / 2 * noise_ax, 0, d_t2 * noise_ax, 0,
         0, d_t3 / 2 * noise_ay, 0, d_t2 * noise_ay;



  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar Updates
    Tools t;
    // 1. Update Kalman filter variables H and R for radar
    // Hj_radar_ needs to be computed
    ekf_.H_ = t.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    // 3. Update Using Extended Kalman Filter
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  } else {
    // Laser updates
    // 1. Update Kalman filter variables H and R for laser
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    // 2. Update Using Kalman Filter
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
