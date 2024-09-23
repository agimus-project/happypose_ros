# Class id for which symmetries are considered
string class_id
# Lists discrete and continuous symmetries of considered object.
# If no symmetries of a given type, list is left empty.

# In HappyPose discrete symmetries are represented as
# transformation matrices. Those matrices are directly
# converted to the Transform message.
geometry_msgs/Transform[] symmetries_discrete
ContinuousSymmetry[] symmetries_continuous
