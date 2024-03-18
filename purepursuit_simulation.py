import math
import matplotlib.pyplot as plt
import numpy as np
import time

# Vehicle parameters (m)
LENGTH = 4.5
WIDTH = 2.0
BACKTOWHEEL = 1.0
WHEEL_LEN = 0.3
WHEEL_WIDTH = 0.2
TREAD = 0.7
WB = 2.675

# plotting function for simulation
def plotVehicle(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):
    """
    The function is to plot the vehicle
    it is copied from https://github.com/AtsushiSakai/PythonRobotics/blob/187b6aa35f3cbdeca587c0abdb177adddefc5c2a/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py#L109
    """
    outline = np.array(
        [
            [
                -BACKTOWHEEL,
                (LENGTH - BACKTOWHEEL),
                (LENGTH - BACKTOWHEEL),
                -BACKTOWHEEL,
                -BACKTOWHEEL,
            ],
            [WIDTH / 2, WIDTH / 2, -WIDTH / 2, -WIDTH / 2, WIDTH / 2],
        ]
    )

    fr_wheel = np.array(
        [
            [WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
            [
                -WHEEL_WIDTH - TREAD,
                -WHEEL_WIDTH - TREAD,
                WHEEL_WIDTH - TREAD,
                WHEEL_WIDTH - TREAD,
                -WHEEL_WIDTH - TREAD,
            ],
        ]
    )

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array(
        [[math.cos(steer), math.sin(steer)], [-math.sin(steer), math.cos(steer)]]
    )

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(
        np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), truckcolor
    )
    plt.plot(
        np.array(fr_wheel[0, :]).flatten(),
        np.array(fr_wheel[1, :]).flatten(),
        truckcolor,
    )
    plt.plot(
        np.array(rr_wheel[0, :]).flatten(),
        np.array(rr_wheel[1, :]).flatten(),
        truckcolor,
    )
    plt.plot(
        np.array(fl_wheel[0, :]).flatten(),
        np.array(fl_wheel[1, :]).flatten(),
        truckcolor,
    )
    plt.plot(
        np.array(rl_wheel[0, :]).flatten(),
        np.array(rl_wheel[1, :]).flatten(),
        truckcolor,
    )
    plt.plot(x, y, "*")

# Distance function for simulation
def getDistance(p1, p2):
    """
    Calculate distance
    :param p1: list, point1
    :param p2: list, point2
    :return: float, distance
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)

# Vehicle model for simulation
class Vehicle:
    def __init__(self, x, y, yaw, vel=0):
        """
        Define a vehicle class
        :param x: float, x position
        :param y: float, y position
        :param yaw: float, vehicle heading east of north
        :param vel: float, velocity
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vel = vel
        self.dt = 0.1

    def update(self, command):
        """
        Vehicle motion model, here we are using simple bycicle model
        :param acc: float, acceleration
        :param delta: float, heading control
        """
        delta = command.steer
        acc = command.accel

        wheelsteer = -1 * np.deg2rad(delta/16.8)
        self.yaw += self.vel * math.tan((wheelsteer)) / WB * self.dt
        self.x += self.vel * math.cos(self.yaw) * self.dt
        self.y += self.vel * math.sin(self.yaw) * self.dt
        self.vel += acc/100 * self.dt    # The factor of 1/100 is to convert torque to accelertion.  Since we aren't concerened with velocity control this is good enough

class VehicleCommand:
    def __init__(self):
        self.accel = 0
        self.steer = 0

"""
Controller goes here. Controller can access all of the parameters in the vehicle class to control the car
"""
class PurePursuitController:
    def __init__(self, vehicle):
        self.freqs = 10
        self.vehicle = vehicle
        self.vehicle.dt = 1/self.freqs

        # Current vehicle pose
        self.xc = 0                 # Current Vehicle Easting
        self.yc = 0                 # Current Vehicle Northing
        self.upc = 0                # Current Up (Height)
        self.velocityc = 0          # Current velocity in meters/second
        self.vehicle_headingc = 0   # Vehicle Heading in Degrees East of North

        # Set a lookahead distance, one of the most important components of pure pursuit.
        self.LOOKAHEAD_hard = 7

        # Vehicle Information
        self.WB = 2.675                             # Wheelbase in meters
        self.steer_ratio = 16.8                     # Steering Ratio
        self.max_steer = 500                        # Maximum steering wheel turn
        self.prev_steer_commands = [0, 0, 0 , 0, 0] # Initialize steering filter
        self.steer_filter_size = 4                  # Set steering filter size
        self.steer_rate_limit = 400                 # Steering rate limit in degrees per second


        # Waypoint information
        self.trajectory = np.array([[0, 0, 0]])
        self.idx = 0
        self.v_desired = 0
        self.file_name = "" # name of the waypoint info file if importing waypoints from a file

        # Velocity PID Info. Parameters need to be set in ros before running code
        self.kp = 350
        self.ki = 3
        self.kd = 0
        self.v_error = 0
        self.v_prev_error = 0
        self.integral = 0

        
        # Define torque and steering message as a VehicleCommand()
        self.result = VehicleCommand()
        
    def find_distance_meters(self, x1, y1, x2, y2):
        '''
        Finds the distance between two points
        '''

        # Differences in coordinates
        dx = x2 - x1
        dy = y2 - y1

        distance = np.sqrt(dx**2 + dy**2)
        return distance

    def find_distance(self, x1, y1):
        """
        Determine distance between current position and x1, y1.
        """
        distance = self.find_distance_meters(x1, y1, self.vehicle.x, self.vehicle.y)
        return distance

    def find_distance_index_based(self, idx):
        """
        Determine distance between current position and the waypoint associated with index idx. 
        """
        x1 = float(self.trajectory[idx][0])
        y1 = float(self.trajectory[idx][1])
        distance = self.find_distance_meters(x1, y1, self.vehicle.x, self.vehicle.y)
        return distance

    def find_nearest_waypoint(self):
        """
        Finds the absolute closest waypoint to the vehicle.  Mostly used to determine what speed the vehicle should be going at. 
        """
        curr_xy = np.array([self.vehicle.x, self.vehicle.y])
        waypoints_xy = self.trajectory[:, :2]
        nearest_idx = np.argmin(np.sum((curr_xy - waypoints_xy)**2, axis=1))
        return nearest_idx

    def idx_close_to_lookahead(self, idx):
        """
        Get closest index to lookahead that is greater than the lookahead.  Only looks for points greater than the idx given to ensure we don't go 
        backwards 
        """
        while self.find_distance_index_based(idx) < self.LOOKAHEAD_hard:
            idx += 1 
        return idx

    def calculate_bearing(self, x1, y1, x2, y2):
        """
        Calculate the angle (bearing) between two points.

        Parameters:
        - x1, y1: Easting and Northing of the first point (in meters).
        - x2, y2: Easting and Northing of the second point (in meters).

        Returns:
        - bearing: Angle in degrees counterclockwise of east*** 
        ***Note: This bearing is different from the one we use on the car, and so the calculation is different
        """
        dx = x2 - x1
        dy = y2 - y1

        # Calculate bearing in radians counterclockwise from the positive x-axis
        bearing_rad = np.arctan2(dy, dx)

        # Convert bearing from radians to degrees
        bearing_deg = np.degrees(bearing_rad)

        # Normalize bearing to be in the range [0, 360)
        bearing_deg = (bearing_deg + 360) % 360

        return bearing_deg
    
        
    def filter_steering(self, new_command):
        """
        Simple moving average filter for steering commands
        """
        if len(self.prev_steer_commands) >= self.steer_filter_size:
            window = self.prev_steer_commands[-self.steer_filter_size:]
        else:
            window = self.prev_steer_commands
        
        window.append(new_command)
        output = np.average(window)
        self.prev_steer_commands = window
        return output
    
    def rate_limit_steering(self, new_command):
        """
        Applies the rate limit to the steering command
        """
        max_change = self.steer_rate_limit / self.freqs  # Convert maximum rate to per iteration
        change = new_command - self.prev_steer_commands[-1]
        if abs(change) > max_change:
            if change > 0:
                return self.prev_steer_commands[-1] + max_change
            else:
                return self.prev_steer_commands[-1] - max_change
        return new_command


    def follow_trajectory(self):
        """
        Use all of the other functions to run a pure pursuit controller.  There is also a velocity PID controller in here. 
        """

        # Plotting Data (Just for sim)
        traj_ego_x = []
        traj_ego_y = []
        plt.figure(figsize=(12, 8))

        # Set goal (Just for sim)
        goal = self.trajectory[-1][1:]

        count = 0 # counter to limit plotting

        print("STARTING PURE PURSUIT")
        while getDistance([self.vehicle.x, self.vehicle.y], goal) > 1: # continuously run until the distance between the vehicle and goal is less than 1m
            nearest_idx = self.find_nearest_waypoint()

            # Find the waypoint closest to one lookahead distance away from the vehicle
            idx_near_lookahead = self.idx_close_to_lookahead(nearest_idx)
 
            # Set the target point to the point closest to one lookahead distance from the vehicle
            target_x = float(self.trajectory[idx_near_lookahead][0])
            target_y = float(self.trajectory[idx_near_lookahead][1])

            """
            VELOCITY PID CONTROLLER
            """
            dt = 1.0 / self.freqs

            # Calculate the error in velocity
            self.v_desired = float(self.trajectory[nearest_idx][2])
            self.v_error = self.v_desired - self.vehicle.vel

            # Calculate each component of the PID control
            P_vel = self.kp * self.v_error
            I_vel = self.integral + self.ki * self.v_error * dt
            D_vel = self.kd * (self.v_error - self.v_prev_error) / dt

            # Torque to be sent is equal to each of the PID components summed
            torque = P_vel + I_vel + D_vel
            self.v_prev_error = self.v_error
            self.integral = I_vel

            """
            PURE PURSUIT CONTROLLER
            """

            # Calculate bearing (angle between current point and next point relative to north)
            bearing = np.deg2rad(self.calculate_bearing(self.vehicle.x, self.vehicle.y, target_x, target_y))

            # calculate difference between current vehicle heading and bearing
            alpha = bearing - self.vehicle.yaw

            # Set the lookahead distance based on target point
            lookahead = self.find_distance(target_x, target_y)

            # Set the new steering angle.  This is a commonly used formula for Pure Pursuit algos.  This is the angle the wheels need to be at.
            steering_angle = np.degrees(np.arctan((2 * self.WB * np.sin(alpha)) / lookahead))

            # Set the steering wheel_angle.  This ratio will need to be tuned. 
            steering_wheel_angle = self.steer_ratio * steering_angle

            # Set max steering wheel turning angle.  The steering rack is limited to +-550
            if steering_wheel_angle > self.max_steer:
                steering_wheel_angle = self.max_steer
            elif steering_wheel_angle < -self.max_steer:
                steering_wheel_angle = -self.max_steer

            steering_wheel_angle = self.rate_limit_steering(steering_wheel_angle)
            steering_wheel_angle = self.filter_steering(steering_wheel_angle)

            """
            The controller deviates from the actual controller below.  In the actual controller we publish the commands
            to the car rather than a simulated car. 
            """
            # Publish controller messages to simulated vehicle
            self.result.accel = round(torque)
            self.result.steer = -1 * steering_wheel_angle

            self.vehicle.update(self.result)

            # Store the overall trajectory for plotting
            traj_ego_x.append(self.vehicle.x)
            traj_ego_y.append(self.vehicle.y)

            # Plot information
            if count % 5 == 0:
                plt.cla()
                plt.plot([[point[0]] for point in self.trajectory], [[point[1]] for point in self.trajectory], "-r", linewidth=5, label="course")
                plt.plot(traj_ego_x, traj_ego_y, "-b", linewidth=2, label="trajectory")
                plt.plot(target_x, target_y, "og", ms=5, label="target point")
                plotVehicle(self.vehicle.x, self.vehicle.y, self.vehicle.yaw, self.result.steer/16.8)
                plt.xlabel("x[m]")
                plt.ylabel("y[m]")
                plt.axis("equal")
                plt.legend()
                plt.grid(True)
                plt.pause(0.5)
            count += 1

            time.sleep(1/self.freqs)


def main():
    # Create vehicle Vehicle(starting easting (meters), starting northing (meters), starting angle (radians))
    ego = Vehicle(7, 10, np.deg2rad(45))

    controller = PurePursuitController(ego)
    
    # Insert vehicle trajectory as an np array
    controller.trajectory = np.array([[8.817279340795597, 8.951623338550096, 3], [9.243446307196033, 9.247059590829299, 3], [9.770836994531304, 9.612811602208538, 3], [10.384637342409935, 10.038718925218145, 3], [11.06344454672195, 10.510087695359264, 3], [11.783495979478916, 11.010627306905647, 3], [12.523366697011081, 11.5257003781365, 3], [13.268210048553824, 12.045229987720347, 3], [14.012695671613034, 12.56570323923631, 3], [14.756778536902022, 13.087239647399311, 3], [15.500415145754134, 13.609955115213655, 3], [16.243558234577577, 14.133974914316626, 3], [16.986158457010852, 14.65942947592572, 3], [17.72816646816028, 15.186449237418547, 3], [18.46953291994575, 15.715164635565605, 3], [19.21020845885604, 16.24570611080174, 3], [19.95014374117499, 16.77820410689574, 3], [20.68928941199441, 17.31278906756797, 3], [21.427596111984172, 17.84959144905835, 3], [22.165014318107254, 18.388742104971378, 3], [22.901493431243807, 18.930374394073198, 3], [23.636979731405496, 19.474628882495793, 3], [24.371414466277542, 20.021657207966527, 3], [25.104735301633454, 20.571617220975114, 3], [25.836882724846596, 21.124655532070943, 3], [26.567810185359967, 21.68088125025958, 3], [27.297495210136198, 22.240337887619006, 3], [28.02594905412427, 22.80297977853867, 3], [28.75322288056136, 23.368657751871517, 3], [29.479409237474563, 23.937116853155928, 3], [30.204638451146074, 24.508006522691105, 3], [30.929070647964764, 25.0809011635014, 3], [31.65288491817531, 25.655327026516076, 3], [32.37626809477842, 26.230789725419605, 3], [33.099405939058315, 26.806795645855516, 3], [33.82247923838523, 27.38286173712232, 3], [34.5456645620645, 27.958514227936554, 3], [35.26913659406617, 28.533283446344406, 3], [35.993069596590566, 29.106700674579407, 3], [36.717637818335184, 29.67829721460615, 3], [37.44301552853629, 30.247604345795935, 3], [38.16937702833274, 30.81415330685001, 3], [38.896896644777456, 31.37747528225812, 3], [39.625748740027404, 31.93710139547364, 3], [40.35610771725785, 32.49256271106202, 3], [41.08814802075512, 33.04339020065491, 3], [41.822044142760994, 33.589114747541714, 3], [42.55797206462771, 34.12926367025344, 3], [43.29612323542947, 34.663326590085944, 3], 
[44.03674728638443, 35.1906466733334, 3], [44.78020185230943, 35.71028366724545, 2], [45.52687236875571, 36.221219753333045, 2], [46.276797060560554, 36.723408264480234, 2], [47.0291243723515, 37.21939781677928, 2], [47.78199702161366, 37.71470143130165, 2], [48.533402071935875, 38.21512582668033, 2], [49.28275137223078, 38.72208548514966, 2], [50.03221116559717, 39.22907848463311, 2], [50.78685021818065, 39.721122967271825, 2], [51.55342289300921, 40.17643197759686, 2], [52.33815163150277, 40.5691860778823, 2], [53.14410405323956, 40.87332544092324, 2], [53.969261453782366, 41.06774186003505, 2], [54.806772170660416, 41.14195231988606, 2], [55.64773947183188, 41.09918218405722, 2], [56.48455432329441, 40.95402399003602, 2], [57.31236305209174, 40.72552930465406, 2], [58.12835934795435, 40.42912863910186, 2], [58.92946273393275, 40.07027793728594, 2], 
[59.70934328833897, 39.642473741610075, 2], [60.45719125270655, 39.1314717044911, 2], [61.15912325269102, 38.52318368253758, 2], [61.80003367813307, 37.80968665332237, 2], [62.364552116141745, 36.99131362083416, 2], [62.83844882654754, 36.07671661684997, 2], [63.21053148835779, 35.08160293967791, 2], [63.47268568710703, 34.026043552073126, 2], [63.6179120764124, 32.93212910698313, 2], [63.63910041389571, 31.823590714720442, 2], 
[63.530333997225924, 30.726500867630413, 2], [63.29015021968082, 29.66860860642374, 2], [62.924953334630395, 28.675527549449427, 2], [62.45054348413521, 27.764507912681534, 2], [61.88973736254987, 26.93983984429854, 2], [61.266486677836426, 26.19390960346604, 2], [60.60070083214909, 25.512392500163834, 2], [59.90709867731267, 24.878985217664674, 2], [59.196504097339364, 24.278332188265853, 2], [58.47638369606908, 23.698323614125307, 2], [57.75072877170913, 23.131574447679757, 2], [57.02092372925606, 22.57497783606107, 2], [56.28716310683364, 22.027864187070357, 2], [55.5493586178404, 21.49027996115255, 2], [54.807548999843576, 20.962039397855488, 2], [54.06202606857368, 20.442430889773647, 2], [53.31326963928182, 19.93033257918949, 5], [52.561834856432256, 19.424447471919557, 5], [51.80829422562828, 18.923436208279902, 5], [51.05322802653256, 18.425939167770068, 5], [50.29722980881575, 17.930561475117223, 5], [49.54090453567499, 17.435876785470324, 5], [48.784851108981925, 16.94047197723167, 5], [48.029631546514246, 16.44302666702866, 5], [47.27573447843321, 15.942405988563344, 5], [46.523540944101576, 15.437744141199797, 5], [45.77329913096197, 14.928501325869302, 5], [45.025113181003476, 14.414482739224843, 5], [44.27894879252612, 13.895816118649504, 5], [43.53465511380603, 13.372892577297108, 5], [42.79199880791975, 12.846282733814222, 5], [42.05070317110959, 12.316645293611362, 5], [41.31048261601092, 11.784649648691921, 5], [40.57106285731606, 11.250932873289617, 5], [39.83218369384324, 10.716096493575613, 5], [39.09359258151875, 10.180723287445026, 5], [38.355039165874345, 9.645390703087386, 5], [37.616273346599314, 9.110675584846016, 5], [36.877045007694676, 8.577154804907343, 5], [36.13710402575747, 8.04540526571037, 5], [35.39620024710329, 7.516003933977169, 5], [34.65408340192009, 6.989528085941537, 5], [33.91050113308692, 6.466560214140055, 5], [33.165191001893234, 5.947708451940178, 5], [32.41786640628844, 5.4336446560123335, 5], [31.668211914613465, 4.925124689684181, 
5], [30.915910393716974, 4.422935775718956, 5], [30.16070536966654, 3.9277561166212536, 5], [29.402481041784455, 3.439960809881865, 5], [28.64133751443803, 2.9594228503160704, 5], [27.87764381423303, 2.4853522748347516, 5], [27.1120589840853, 2.0162060591996935, 4.5], [26.345520216000917, 1.5496853338187597, 4.5], [25.57920800161191, 1.0828075414753384, 4.5], [24.81450472005138, 0.6120186539710464, 4.5], [24.05295453991904, 0.13333450975759603, 4.5], [23.296210727185105, -0.3574340970905563, 4.5], [22.545949094204158, -0.8643216708861694, 4.5], [21.803741413013768, -1.3907521746687834, 4.5], [21.070899230502956, -1.9391038021329403, 4.5], [20.348311349997257, -2.5104498674301166, 4.5], [19.63630322077613, -3.104573569683249, 4.5], [18.93461115611139, -3.7202997599484764, 4.5], [18.24271366312916, -4.356250901688506, 4.5], [17.5606898391306, -5.012118459438171, 4.5], [16.890469353450282, -5.690252750685524, 4.5], [16.237027437715593, -6.396879715086014, 4.5], [15.609081370647765, -7.142117219092166, 4.5], [15.019180837572433, -7.938395305435593, 4.5], [14.48318790050328, -8.79740375473639, 4.5], [14.018930559206401, -9.726254642993677, 4.5], [13.643773750327345, -10.72420311474879, 4.5], [13.371433061043057, -11.78155388105482, 4.5], [13.209277192577176, -12.88158886109488, 4.5], [13.157590283826814, -14.00452856885206, 4.5], [13.211211323641812, -15.131279773743948, 4.5], [13.362452214163088, -16.245299026649185, 4.5], [13.603570606164398, -17.3326121546293, 4.5], [13.927815854933698, -18.381135596428763, 4.5], [14.32921703266775, -19.380204661232625, 4.5], 
[14.801889236311396, -20.32058156446096, 4.5], [15.339505337991154, -21.194847953598043, 4.5], [15.9351070985806, -21.99792083926008, 4.5], [16.58121759125531, -22.727439715860992, 4.5], [17.270194189152836, -23.38377851107781, 4.5], [17.99470570917413, -23.969490925177134, 4.5], [18.748170681081966, -24.48823086111385, 4.5], [19.525029715488746, -24.94342230952577, 4.5], [20.320798419691208, -25.337062538127924, 4.5], [21.131908136745253, -25.669068913329234, 4.5], [21.955396051799042, -25.93741689898303, 4.5], [22.788554280428247, -26.138953214502664, 4.5], [23.628656860081144, -26.270509910544632, 4.5], [24.47284310355012, -26.329963462351273, 4.5], [25.31817511587802, -26.3169860970986, 4.5], [26.161817388769645, -26.233324743484282, 4.5], [27.001228046575964, -26.082608264850705, 4.5], [27.834267266605575, -25.869850854883275, 4.5], [28.65921115953134, -25.600837797595915, 4.5], [29.474708985004153, -25.28153000393208, 4.5], [30.279730536010526, -24.91760858158423, 4.5], [31.073548760207498, -24.514248570737173, 4.5], [31.855773062451114, -24.076116044029, 4.5], [32.626403073973094, -23.607468856144195, 4.5], [33.38585609538078, 
-23.1122501805647, 4.5], [34.13494577496382, -22.594156396145593, 4.5], [34.87481773990061, -22.056698326689563, 4.5], [35.60686233534966, -21.503270386452602, 4.5], [36.33262959959785, -20.937231058271646, 4.5], [37.05376359116933, -20.361975745185184, 4.5], [37.77194972389284, -19.780948378989578, 4.5], [38.48885289928303, -19.197556861250767, 4.5], [39.20603255348405, -18.615016277273117, 4.5], [39.924845760846644, -18.03616530848971, 4.5], [40.64636293254814, -17.463296134251312, 4.5], [41.37131534871523, -16.898031623260085, 4.5], [42.10008515949712, -16.341275225008403, 4.5], [42.83273824376491, -15.793245972736486, 4.5], [43.569091119623145, -15.253593954334079, 4.5], [44.30879718331391, -14.721574370819422, 4.5], [45.05143459448101, -14.196243590444205, 4.5], [45.7965767840537, -13.676628575841885, 7], [46.5438291800095, -13.16182265429077, 7], [47.29282981878376, -12.650994356459167, 7], [48.04323325904743, -12.14335209232797, 7], [48.794698643422, -11.638115662748117, 7], [49.54688565911357, -11.1345062778926, 7], [50.29945402333785, -10.631745242634137, 7], [51.05206346948415, -10.129053923247518, 7], [51.80437341780892, -9.625652900195599, 7], [52.55604135797401, -9.120757792475334, 7], [53.30671912542761, -8.613570039898462, 7], [54.056049643357376, -8.103269285110988, 7], [54.803670106117266, -7.589022326376508, 7], [55.5492248187962, -7.070016011655532, 7], [56.292385183100706, -6.545506730272911, 7], 
[57.03287185056039, -6.014873314996316, 7], [57.770474311290165, -5.47766121175534, 7], [58.505064000999624, -4.93360897092378, 7], [59.23659818040686, -4.382651865140565, 7], [59.96511362014514, -3.824902054050696, 7], [60.69071106022009, -3.260609416381992, 7], [61.41353350404696, -2.6901108958288518, 7], [62.13374303031407, -2.11377918216355, 7], [62.851502145555436, -1.531983327543369, 7], [63.56696484391371, -0.9450713979257209, 7], [64.28027725317645, -0.3533736154631012, 7], [64.99158193863038, 0.24278763482373628, 7], [65.70102060541211, 0.843091783336492, 7], [66.40873488757583, 1.4472184242602264, 7], [67.11486640513994, 2.0548471885215744, 7], [67.81955675387623, 2.6656577442533127, 7], [68.52294751704311, 3.279329793702873, 7], [69.22518025219625, 3.895543066260153, 7], [69.926396514361, 4.513977319010332, 7], [70.62673783572816, 5.134312329381363, 7], [71.3263457465505, 5.756227886509178, 7], [72.02536190269942, 6.379403534466839, 7], [72.72392880775288, 7.003517139186424, 7], [73.42219149664922, 7.628241702933638, 7], [74.12029894203152, 8.25324269973038, 7], [74.8184027679058, 8.878181256502964, 0], [75.51665181145654, 
9.502725895513699, 0], [76.21518381313084, 10.126570299330607, 0], [76.91411643937323, 10.74945250749439, 0], [77.61353976087672, 11.371171218094489, 0], [78.31351166679656, 11.991595920950195, 0], [79.01405715955751, 12.610668892957285, 0], [79.7151715970651, 13.228398575409514, 0], [80.41682727814835, 13.84484566157882, 0], [81.1189819819834, 14.460104644866481, 0], [81.8215877080518, 15.074284725605203, 0], [82.52459739729099, 15.68749484720647, 0], [83.22796985775453, 16.299832820867497, 0], [83.93167084991941, 16.91138265169038, 0], [84.63566655762807, 17.522227451873142, 0], [85.33492896027612, 18.12816115011614, 0], [86.01543649426131, 18.717237597706262, 0], [86.65694972046472, 19.272142367955457, 0], [87.23701849193714, 19.773627663960482, 0], [87.7354212511163, 20.20434138851359, 0], [88.13816161653445, 20.552283235774816, 0]])

    controller.follow_trajectory()

if __name__ == "__main__":
    main()