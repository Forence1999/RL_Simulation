??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:R*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:R*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
 
?


kernel
bias
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
w
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api


0
1
 


0
1
?
metrics
layer_metrics
	variables
regularization_losses

layers
layer_regularization_losses
trainable_variables
non_trainable_variables
 
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 


0
1
 


0
1
?
metrics
 layer_metrics
	variables
regularization_losses

!layers
"layer_regularization_losses
trainable_variables
#non_trainable_variables
 
 
 
 
?
$metrics
%layer_metrics
	variables
regularization_losses

&layers
'layer_regularization_losses
trainable_variables
(non_trainable_variables
 
 
 
?
)metrics
*layer_metrics
	variables
regularization_losses

+layers
,layer_regularization_losses
trainable_variables
-non_trainable_variables
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_conv2d_10_inputPlaceholder*/
_output_shapes
:?????????R*
dtype0*$
shape:?????????R
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_10_inputconv2d_10/kernelconv2d_10/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_signature_wrapper_144227217
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_save_144227348
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_10/kernelconv2d_10/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference__traced_restore_144227364??
?
?
'__inference_signature_wrapper_144227217
conv2d_10_input
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__wrapped_model_1442270902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????R
)
_user_specified_nameconv2d_10_input
?
?
D__inference_model_layer_call_and_return_conditional_losses_144227179

inputs
conv2d_10_144227171
conv2d_10_144227173
identity??!conv2d_10/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_144227171conv2d_10_144227173*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_1442271042#
!conv2d_10/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_layer_call_and_return_conditional_losses_1442271322
reshape/PartitionedCall?
softmax/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_softmax_layer_call_and_return_conditional_losses_1442271452
softmax/PartitionedCall?
IdentityIdentity softmax/PartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
~
)__inference_model_layer_call_fn_144227264

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_1442271792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
)__inference_model_layer_call_fn_144227186
conv2d_10_input
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_1442271792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????R
)
_user_specified_nameconv2d_10_input
?
?
$__inference__wrapped_model_144227090
conv2d_10_input2
.model_conv2d_10_conv2d_readvariableop_resource3
/model_conv2d_10_biasadd_readvariableop_resource
identity??&model/conv2d_10/BiasAdd/ReadVariableOp?%model/conv2d_10/Conv2D/ReadVariableOp?
%model/conv2d_10/Conv2D/ReadVariableOpReadVariableOp.model_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02'
%model/conv2d_10/Conv2D/ReadVariableOp?
model/conv2d_10/Conv2DConv2Dconv2d_10_input-model/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model/conv2d_10/Conv2D?
&model/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp/model_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model/conv2d_10/BiasAdd/ReadVariableOp?
model/conv2d_10/BiasAddBiasAddmodel/conv2d_10/Conv2D:output:0.model/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model/conv2d_10/BiasAddz
model/reshape/ShapeShape model/conv2d_10/BiasAdd:output:0*
T0*
_output_shapes
:2
model/reshape/Shape?
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model/reshape/strided_slice/stack?
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#model/reshape/strided_slice/stack_1?
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#model/reshape/strided_slice/stack_2?
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/reshape/strided_slice?
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
model/reshape/Reshape/shape/1?
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
model/reshape/Reshape/shape?
model/reshape/ReshapeReshape model/conv2d_10/BiasAdd:output:0$model/reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
model/reshape/Reshape?
model/softmax/SoftmaxSoftmaxmodel/reshape/Reshape:output:0*
T0*'
_output_shapes
:?????????2
model/softmax/Softmax?
IdentityIdentitymodel/softmax/Softmax:softmax:0'^model/conv2d_10/BiasAdd/ReadVariableOp&^model/conv2d_10/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::2P
&model/conv2d_10/BiasAdd/ReadVariableOp&model/conv2d_10/BiasAdd/ReadVariableOp2N
%model/conv2d_10/Conv2D/ReadVariableOp%model/conv2d_10/Conv2D/ReadVariableOp:` \
/
_output_shapes
:?????????R
)
_user_specified_nameconv2d_10_input
?
?
D__inference_model_layer_call_and_return_conditional_losses_144227154
conv2d_10_input
conv2d_10_144227115
conv2d_10_144227117
identity??!conv2d_10/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputconv2d_10_144227115conv2d_10_144227117*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_1442271042#
!conv2d_10/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_layer_call_and_return_conditional_losses_1442271322
reshape/PartitionedCall?
softmax/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_softmax_layer_call_and_return_conditional_losses_1442271452
softmax/PartitionedCall?
IdentityIdentity softmax/PartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall:` \
/
_output_shapes
:?????????R
)
_user_specified_nameconv2d_10_input
?
b
F__inference_softmax_layer_call_and_return_conditional_losses_144227314

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference__traced_restore_144227364
file_prefix%
!assignvariableop_conv2d_10_kernel%
!assignvariableop_1_conv2d_10_bias

identity_3??AssignVariableOp?AssignVariableOp_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_2?

Identity_3IdentityIdentity_2:output:0^AssignVariableOp^AssignVariableOp_1*
T0*
_output_shapes
: 2

Identity_3"!

identity_3Identity_3:output:0*
_input_shapes

: ::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
b
F__inference_softmax_layer_call_and_return_conditional_losses_144227145

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_model_layer_call_and_return_conditional_losses_144227165
conv2d_10_input
conv2d_10_144227157
conv2d_10_144227159
identity??!conv2d_10/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputconv2d_10_144227157conv2d_10_144227159*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_1442271042#
!conv2d_10/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_layer_call_and_return_conditional_losses_1442271322
reshape/PartitionedCall?
softmax/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_softmax_layer_call_and_return_conditional_losses_1442271452
softmax/PartitionedCall?
IdentityIdentity softmax/PartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall:` \
/
_output_shapes
:?????????R
)
_user_specified_nameconv2d_10_input
?
G
+__inference_softmax_layer_call_fn_144227319

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_softmax_layer_call_and_return_conditional_losses_1442271452
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_model_layer_call_and_return_conditional_losses_144227236

inputs,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource
identity?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_10/BiasAddh
reshape/ShapeShapeconv2d_10/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slice}
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape/Reshape/shape/1?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeconv2d_10/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
reshape/Reshapey
softmax/SoftmaxSoftmaxreshape/Reshape:output:0*
T0*'
_output_shapes
:?????????2
softmax/Softmax?
IdentityIdentitysoftmax/Softmax:softmax:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
)__inference_model_layer_call_fn_144227206
conv2d_10_input
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_1442271992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????R
)
_user_specified_nameconv2d_10_input
?
?
-__inference_conv2d_10_layer_call_fn_144227292

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_1442271042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?	
?
H__inference_conv2d_10_layer_call_and_return_conditional_losses_144227283

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
"__inference__traced_save_144227348
file_prefix/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
: :R:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:R: 

_output_shapes
::

_output_shapes
: 
?
~
)__inference_model_layer_call_fn_144227273

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_1442271992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?	
?
H__inference_conv2d_10_layer_call_and_return_conditional_losses_144227104

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?

b
F__inference_reshape_layer_call_and_return_conditional_losses_144227304

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_model_layer_call_and_return_conditional_losses_144227255

inputs,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource
identity?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_10/BiasAddh
reshape/ShapeShapeconv2d_10/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slice}
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape/Reshape/shape/1?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeconv2d_10/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
reshape/Reshapey
softmax/SoftmaxSoftmaxreshape/Reshape:output:0*
T0*'
_output_shapes
:?????????2
softmax/Softmax?
IdentityIdentitysoftmax/Softmax:softmax:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?

b
F__inference_reshape_layer_call_and_return_conditional_losses_144227132

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_model_layer_call_and_return_conditional_losses_144227199

inputs
conv2d_10_144227191
conv2d_10_144227193
identity??!conv2d_10/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_144227191conv2d_10_144227193*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_1442271042#
!conv2d_10/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_layer_call_and_return_conditional_losses_1442271322
reshape/PartitionedCall?
softmax/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_softmax_layer_call_and_return_conditional_losses_1442271452
softmax/PartitionedCall?
IdentityIdentity softmax/PartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
G
+__inference_reshape_layer_call_fn_144227309

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_layer_call_and_return_conditional_losses_1442271322
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
conv2d_10_input@
!serving_default_conv2d_10_input:0?????????R;
softmax0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?q
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
._default_save_signature
*/&call_and_return_all_conditional_losses
0__call__"?
_tf_keras_network?{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_10_input"}, "name": "conv2d_10_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["conv2d_10_input", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}, "name": "reshape", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "activation": "softmax"}, "name": "softmax", "inbound_nodes": [[["reshape", 0, 0, {}]]]}], "input_layers": [["conv2d_10_input", 0, 0]], "output_layers": [["softmax", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 8, 82]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_10_input"}, "name": "conv2d_10_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["conv2d_10_input", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}, "name": "reshape", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "activation": "softmax"}, "name": "softmax", "inbound_nodes": [[["reshape", 0, 0, {}]]]}], "input_layers": [["conv2d_10_input", 0, 0]], "output_layers": [["softmax", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_10_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_10_input"}}
?



kernel
bias
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
*1&call_and_return_all_conditional_losses
2__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 82}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 8, 82]}}
?
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
*3&call_and_return_all_conditional_losses
4__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*5&call_and_return_all_conditional_losses
6__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "activation": "softmax"}}
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
?
metrics
layer_metrics
	variables
regularization_losses

layers
layer_regularization_losses
trainable_variables
non_trainable_variables
0__call__
._default_save_signature
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
,
7serving_default"
signature_map
*:(R2conv2d_10/kernel
:2conv2d_10/bias
 "
trackable_dict_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
?
metrics
 layer_metrics
	variables
regularization_losses

!layers
"layer_regularization_losses
trainable_variables
#non_trainable_variables
2__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
$metrics
%layer_metrics
	variables
regularization_losses

&layers
'layer_regularization_losses
trainable_variables
(non_trainable_variables
4__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
)metrics
*layer_metrics
	variables
regularization_losses

+layers
,layer_regularization_losses
trainable_variables
-non_trainable_variables
6__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
$__inference__wrapped_model_144227090?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *6?3
1?.
conv2d_10_input?????????R
?2?
D__inference_model_layer_call_and_return_conditional_losses_144227154
D__inference_model_layer_call_and_return_conditional_losses_144227236
D__inference_model_layer_call_and_return_conditional_losses_144227165
D__inference_model_layer_call_and_return_conditional_losses_144227255?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_model_layer_call_fn_144227206
)__inference_model_layer_call_fn_144227273
)__inference_model_layer_call_fn_144227264
)__inference_model_layer_call_fn_144227186?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_conv2d_10_layer_call_and_return_conditional_losses_144227283?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_conv2d_10_layer_call_fn_144227292?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_reshape_layer_call_and_return_conditional_losses_144227304?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_reshape_layer_call_fn_144227309?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_softmax_layer_call_and_return_conditional_losses_144227314?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_softmax_layer_call_fn_144227319?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_signature_wrapper_144227217conv2d_10_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
$__inference__wrapped_model_144227090y
@?=
6?3
1?.
conv2d_10_input?????????R
? "1?.
,
softmax!?
softmax??????????
H__inference_conv2d_10_layer_call_and_return_conditional_losses_144227283l
7?4
-?*
(?%
inputs?????????R
? "-?*
#? 
0?????????
? ?
-__inference_conv2d_10_layer_call_fn_144227292_
7?4
-?*
(?%
inputs?????????R
? " ???????????
D__inference_model_layer_call_and_return_conditional_losses_144227154u
H?E
>?;
1?.
conv2d_10_input?????????R
p

 
? "%?"
?
0?????????
? ?
D__inference_model_layer_call_and_return_conditional_losses_144227165u
H?E
>?;
1?.
conv2d_10_input?????????R
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_layer_call_and_return_conditional_losses_144227236l
??<
5?2
(?%
inputs?????????R
p

 
? "%?"
?
0?????????
? ?
D__inference_model_layer_call_and_return_conditional_losses_144227255l
??<
5?2
(?%
inputs?????????R
p 

 
? "%?"
?
0?????????
? ?
)__inference_model_layer_call_fn_144227186h
H?E
>?;
1?.
conv2d_10_input?????????R
p

 
? "???????????
)__inference_model_layer_call_fn_144227206h
H?E
>?;
1?.
conv2d_10_input?????????R
p 

 
? "???????????
)__inference_model_layer_call_fn_144227264_
??<
5?2
(?%
inputs?????????R
p

 
? "???????????
)__inference_model_layer_call_fn_144227273_
??<
5?2
(?%
inputs?????????R
p 

 
? "???????????
F__inference_reshape_layer_call_and_return_conditional_losses_144227304`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????
? ?
+__inference_reshape_layer_call_fn_144227309S7?4
-?*
(?%
inputs?????????
? "???????????
'__inference_signature_wrapper_144227217?
S?P
? 
I?F
D
conv2d_10_input1?.
conv2d_10_input?????????R"1?.
,
softmax!?
softmax??????????
F__inference_softmax_layer_call_and_return_conditional_losses_144227314X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
+__inference_softmax_layer_call_fn_144227319K/?,
%?"
 ?
inputs?????????
? "??????????