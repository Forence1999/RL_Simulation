??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
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
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
value_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:R*"
shared_namevalue_conv/kernel

%value_conv/kernel/Read/ReadVariableOpReadVariableOpvalue_conv/kernel*&
_output_shapes
:R*
dtype0
v
value_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namevalue_conv/bias
o
#value_conv/bias/Read/ReadVariableOpReadVariableOpvalue_conv/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:R* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:R*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
?
value_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*#
shared_namevalue_dense/kernel
y
&value_dense/kernel/Read/ReadVariableOpReadVariableOpvalue_dense/kernel*
_output_shapes

:*
dtype0
x
value_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namevalue_dense/bias
q
$value_dense/bias/Read/ReadVariableOpReadVariableOpvalue_dense/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer-7
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
R
$regularization_losses
%	variables
&trainable_variables
'	keras_api
R
(regularization_losses
)	variables
*trainable_variables
+	keras_api
R
,regularization_losses
-	variables
.trainable_variables
/	keras_api
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
?

0layers
	regularization_losses

	variables
trainable_variables
1layer_regularization_losses
2layer_metrics
3metrics
4non_trainable_variables
 
][
VARIABLE_VALUEvalue_conv/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEvalue_conv/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

5layers
regularization_losses
	variables
trainable_variables
6layer_regularization_losses
7layer_metrics
8metrics
9non_trainable_variables
 
 
 
?

:layers
regularization_losses
	variables
trainable_variables
;layer_regularization_losses
<layer_metrics
=metrics
>non_trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

?layers
regularization_losses
	variables
trainable_variables
@layer_regularization_losses
Alayer_metrics
Bmetrics
Cnon_trainable_variables
^\
VARIABLE_VALUEvalue_dense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEvalue_dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

Dlayers
 regularization_losses
!	variables
"trainable_variables
Elayer_regularization_losses
Flayer_metrics
Gmetrics
Hnon_trainable_variables
 
 
 
?

Ilayers
$regularization_losses
%	variables
&trainable_variables
Jlayer_regularization_losses
Klayer_metrics
Lmetrics
Mnon_trainable_variables
 
 
 
?

Nlayers
(regularization_losses
)	variables
*trainable_variables
Olayer_regularization_losses
Player_metrics
Qmetrics
Rnon_trainable_variables
 
 
 
?

Slayers
,regularization_losses
-	variables
.trainable_variables
Tlayer_regularization_losses
Ulayer_metrics
Vmetrics
Wnon_trainable_variables
8
0
1
2
3
4
5
6
7
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
 
 
 
 
 
?
serving_default_feature_inputPlaceholder*/
_output_shapes
:?????????R*
dtype0*$
shape:?????????R
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_feature_inputvalue_conv/kernelvalue_conv/biasconv2d_1/kernelconv2d_1/biasvalue_dense/kernelvalue_dense/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_37704868
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%value_conv/kernel/Read/ReadVariableOp#value_conv/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp&value_dense/kernel/Read/ReadVariableOp$value_dense/bias/Read/ReadVariableOpConst*
Tin

2*
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
GPU 2J 8? **
f%R#
!__inference__traced_save_37705150
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamevalue_conv/kernelvalue_conv/biasconv2d_1/kernelconv2d_1/biasvalue_dense/kernelvalue_dense/bias*
Tin
	2*
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_37705178??
?=
?
#__inference__wrapped_model_37704582
feature_input5
1model_1_value_conv_conv2d_readvariableop_resource6
2model_1_value_conv_biasadd_readvariableop_resource3
/model_1_conv2d_1_conv2d_readvariableop_resource4
0model_1_conv2d_1_biasadd_readvariableop_resource6
2model_1_value_dense_matmul_readvariableop_resource7
3model_1_value_dense_biasadd_readvariableop_resource
identity??'model_1/conv2d_1/BiasAdd/ReadVariableOp?&model_1/conv2d_1/Conv2D/ReadVariableOp?)model_1/value_conv/BiasAdd/ReadVariableOp?(model_1/value_conv/Conv2D/ReadVariableOp?*model_1/value_dense/BiasAdd/ReadVariableOp?)model_1/value_dense/MatMul/ReadVariableOp?
(model_1/value_conv/Conv2D/ReadVariableOpReadVariableOp1model_1_value_conv_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02*
(model_1/value_conv/Conv2D/ReadVariableOp?
model_1/value_conv/Conv2DConv2Dfeature_input0model_1/value_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_1/value_conv/Conv2D?
)model_1/value_conv/BiasAdd/ReadVariableOpReadVariableOp2model_1_value_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_1/value_conv/BiasAdd/ReadVariableOp?
model_1/value_conv/BiasAddBiasAdd"model_1/value_conv/Conv2D:output:01model_1/value_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_1/value_conv/BiasAdd?
&model_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02(
&model_1/conv2d_1/Conv2D/ReadVariableOp?
model_1/conv2d_1/Conv2DConv2Dfeature_input.model_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_1/conv2d_1/Conv2D?
'model_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/conv2d_1/BiasAdd/ReadVariableOp?
model_1/conv2d_1/BiasAddBiasAdd model_1/conv2d_1/Conv2D:output:0/model_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_1/conv2d_1/BiasAdd?
model_1/value_reshape/ShapeShape#model_1/value_conv/BiasAdd:output:0*
T0*
_output_shapes
:2
model_1/value_reshape/Shape?
)model_1/value_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)model_1/value_reshape/strided_slice/stack?
+model_1/value_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+model_1/value_reshape/strided_slice/stack_1?
+model_1/value_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+model_1/value_reshape/strided_slice/stack_2?
#model_1/value_reshape/strided_sliceStridedSlice$model_1/value_reshape/Shape:output:02model_1/value_reshape/strided_slice/stack:output:04model_1/value_reshape/strided_slice/stack_1:output:04model_1/value_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#model_1/value_reshape/strided_slice?
%model_1/value_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%model_1/value_reshape/Reshape/shape/1?
#model_1/value_reshape/Reshape/shapePack,model_1/value_reshape/strided_slice:output:0.model_1/value_reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#model_1/value_reshape/Reshape/shape?
model_1/value_reshape/ReshapeReshape#model_1/value_conv/BiasAdd:output:0,model_1/value_reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
model_1/value_reshape/Reshape?
)model_1/value_dense/MatMul/ReadVariableOpReadVariableOp2model_1_value_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)model_1/value_dense/MatMul/ReadVariableOp?
model_1/value_dense/MatMulMatMul&model_1/value_reshape/Reshape:output:01model_1/value_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/value_dense/MatMul?
*model_1/value_dense/BiasAdd/ReadVariableOpReadVariableOp3model_1_value_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*model_1/value_dense/BiasAdd/ReadVariableOp?
model_1/value_dense/BiasAddBiasAdd$model_1/value_dense/MatMul:product:02model_1/value_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/value_dense/BiasAdd?
model_1/reshape_1/ShapeShape!model_1/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
model_1/reshape_1/Shape?
%model_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_1/reshape_1/strided_slice/stack?
'model_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_1/reshape_1/strided_slice/stack_1?
'model_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_1/reshape_1/strided_slice/stack_2?
model_1/reshape_1/strided_sliceStridedSlice model_1/reshape_1/Shape:output:0.model_1/reshape_1/strided_slice/stack:output:00model_1/reshape_1/strided_slice/stack_1:output:00model_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model_1/reshape_1/strided_slice?
!model_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!model_1/reshape_1/Reshape/shape/1?
model_1/reshape_1/Reshape/shapePack(model_1/reshape_1/strided_slice:output:0*model_1/reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
model_1/reshape_1/Reshape/shape?
model_1/reshape_1/ReshapeReshape!model_1/conv2d_1/BiasAdd:output:0(model_1/reshape_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
model_1/reshape_1/Reshape?
model_1/add_1/addAddV2$model_1/value_dense/BiasAdd:output:0"model_1/reshape_1/Reshape:output:0*
T0*'
_output_shapes
:?????????2
model_1/add_1/add}
model_1/lambda_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ?E2
model_1/lambda_1/truediv/y?
model_1/lambda_1/truedivRealDivmodel_1/add_1/add:z:0#model_1/lambda_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
model_1/lambda_1/truediv?
IdentityIdentitymodel_1/lambda_1/truediv:z:0(^model_1/conv2d_1/BiasAdd/ReadVariableOp'^model_1/conv2d_1/Conv2D/ReadVariableOp*^model_1/value_conv/BiasAdd/ReadVariableOp)^model_1/value_conv/Conv2D/ReadVariableOp+^model_1/value_dense/BiasAdd/ReadVariableOp*^model_1/value_dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R::::::2R
'model_1/conv2d_1/BiasAdd/ReadVariableOp'model_1/conv2d_1/BiasAdd/ReadVariableOp2P
&model_1/conv2d_1/Conv2D/ReadVariableOp&model_1/conv2d_1/Conv2D/ReadVariableOp2V
)model_1/value_conv/BiasAdd/ReadVariableOp)model_1/value_conv/BiasAdd/ReadVariableOp2T
(model_1/value_conv/Conv2D/ReadVariableOp(model_1/value_conv/Conv2D/ReadVariableOp2X
*model_1/value_dense/BiasAdd/ReadVariableOp*model_1/value_dense/BiasAdd/ReadVariableOp2V
)model_1/value_dense/MatMul/ReadVariableOp)model_1/value_dense/MatMul/ReadVariableOp:^ Z
/
_output_shapes
:?????????R
'
_user_specified_namefeature_input
?
?
.__inference_value_dense_layer_call_fn_37705058

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_value_dense_layer_call_and_return_conditional_losses_377046682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_model_1_layer_call_and_return_conditional_losses_37704768
feature_input
value_conv_37704748
value_conv_37704750
conv2d_1_37704753
conv2d_1_37704755
value_dense_37704759
value_dense_37704761
identity?? conv2d_1/StatefulPartitionedCall?"value_conv/StatefulPartitionedCall?#value_dense/StatefulPartitionedCall?
"value_conv/StatefulPartitionedCallStatefulPartitionedCallfeature_inputvalue_conv_37704748value_conv_37704750*
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
H__inference_value_conv_layer_call_and_return_conditional_losses_377045962$
"value_conv/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallfeature_inputconv2d_1_37704753conv2d_1_37704755*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_377046222"
 conv2d_1/StatefulPartitionedCall?
value_reshape/PartitionedCallPartitionedCall+value_conv/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_value_reshape_layer_call_and_return_conditional_losses_377046502
value_reshape/PartitionedCall?
#value_dense/StatefulPartitionedCallStatefulPartitionedCall&value_reshape/PartitionedCall:output:0value_dense_37704759value_dense_37704761*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_value_dense_layer_call_and_return_conditional_losses_377046682%
#value_dense/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_377046962
reshape_1/PartitionedCall?
add_1/PartitionedCallPartitionedCall,value_dense/StatefulPartitionedCall:output:0"reshape_1/PartitionedCall:output:0*
Tin
2*
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
GPU 2J 8? *L
fGRE
C__inference_add_1_layer_call_and_return_conditional_losses_377047102
add_1/PartitionedCall?
lambda_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
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
F__inference_lambda_1_layer_call_and_return_conditional_losses_377047312
lambda_1/PartitionedCall?
IdentityIdentity!lambda_1/PartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall#^value_conv/StatefulPartitionedCall$^value_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2H
"value_conv/StatefulPartitionedCall"value_conv/StatefulPartitionedCall2J
#value_dense/StatefulPartitionedCall#value_dense/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????R
'
_user_specified_namefeature_input
?
?
*__inference_model_1_layer_call_fn_37704984

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_377048342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?

g
K__inference_value_reshape_layer_call_and_return_conditional_losses_37704650

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
?
E__inference_model_1_layer_call_and_return_conditional_losses_37704834

inputs
value_conv_37704814
value_conv_37704816
conv2d_1_37704819
conv2d_1_37704821
value_dense_37704825
value_dense_37704827
identity?? conv2d_1/StatefulPartitionedCall?"value_conv/StatefulPartitionedCall?#value_dense/StatefulPartitionedCall?
"value_conv/StatefulPartitionedCallStatefulPartitionedCallinputsvalue_conv_37704814value_conv_37704816*
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
H__inference_value_conv_layer_call_and_return_conditional_losses_377045962$
"value_conv/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_37704819conv2d_1_37704821*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_377046222"
 conv2d_1/StatefulPartitionedCall?
value_reshape/PartitionedCallPartitionedCall+value_conv/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_value_reshape_layer_call_and_return_conditional_losses_377046502
value_reshape/PartitionedCall?
#value_dense/StatefulPartitionedCallStatefulPartitionedCall&value_reshape/PartitionedCall:output:0value_dense_37704825value_dense_37704827*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_value_dense_layer_call_and_return_conditional_losses_377046682%
#value_dense/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_377046962
reshape_1/PartitionedCall?
add_1/PartitionedCallPartitionedCall,value_dense/StatefulPartitionedCall:output:0"reshape_1/PartitionedCall:output:0*
Tin
2*
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
GPU 2J 8? *L
fGRE
C__inference_add_1_layer_call_and_return_conditional_losses_377047102
add_1/PartitionedCall?
lambda_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
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
F__inference_lambda_1_layer_call_and_return_conditional_losses_377047312
lambda_1/PartitionedCall?
IdentityIdentity!lambda_1/PartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall#^value_conv/StatefulPartitionedCall$^value_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2H
"value_conv/StatefulPartitionedCall"value_conv/StatefulPartitionedCall2J
#value_dense/StatefulPartitionedCall#value_dense/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
T
(__inference_add_1_layer_call_fn_37705087
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
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
GPU 2J 8? *L
fGRE
C__inference_add_1_layer_call_and_return_conditional_losses_377047102
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_37704731

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ?E2
	truediv/yk
truedivRealDivinputstruediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truediv_
IdentityIdentitytruediv:z:0*
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
?
?
-__inference_value_conv_layer_call_fn_37705003

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
H__inference_value_conv_layer_call_and_return_conditional_losses_377045962
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
?
E__inference_model_1_layer_call_and_return_conditional_losses_37704794

inputs
value_conv_37704774
value_conv_37704776
conv2d_1_37704779
conv2d_1_37704781
value_dense_37704785
value_dense_37704787
identity?? conv2d_1/StatefulPartitionedCall?"value_conv/StatefulPartitionedCall?#value_dense/StatefulPartitionedCall?
"value_conv/StatefulPartitionedCallStatefulPartitionedCallinputsvalue_conv_37704774value_conv_37704776*
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
H__inference_value_conv_layer_call_and_return_conditional_losses_377045962$
"value_conv/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_37704779conv2d_1_37704781*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_377046222"
 conv2d_1/StatefulPartitionedCall?
value_reshape/PartitionedCallPartitionedCall+value_conv/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_value_reshape_layer_call_and_return_conditional_losses_377046502
value_reshape/PartitionedCall?
#value_dense/StatefulPartitionedCallStatefulPartitionedCall&value_reshape/PartitionedCall:output:0value_dense_37704785value_dense_37704787*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_value_dense_layer_call_and_return_conditional_losses_377046682%
#value_dense/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_377046962
reshape_1/PartitionedCall?
add_1/PartitionedCallPartitionedCall,value_dense/StatefulPartitionedCall:output:0"reshape_1/PartitionedCall:output:0*
Tin
2*
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
GPU 2J 8? *L
fGRE
C__inference_add_1_layer_call_and_return_conditional_losses_377047102
add_1/PartitionedCall?
lambda_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
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
F__inference_lambda_1_layer_call_and_return_conditional_losses_377047252
lambda_1/PartitionedCall?
IdentityIdentity!lambda_1/PartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall#^value_conv/StatefulPartitionedCall$^value_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2H
"value_conv/StatefulPartitionedCall"value_conv/StatefulPartitionedCall2J
#value_dense/StatefulPartitionedCall#value_dense/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?	
?
I__inference_value_dense_layer_call_and_return_conditional_losses_37705049

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

c
G__inference_reshape_1_layer_call_and_return_conditional_losses_37704696

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

c
G__inference_reshape_1_layer_call_and_return_conditional_losses_37705070

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
?
E__inference_model_1_layer_call_and_return_conditional_losses_37704745
feature_input
value_conv_37704607
value_conv_37704609
conv2d_1_37704633
conv2d_1_37704635
value_dense_37704679
value_dense_37704681
identity?? conv2d_1/StatefulPartitionedCall?"value_conv/StatefulPartitionedCall?#value_dense/StatefulPartitionedCall?
"value_conv/StatefulPartitionedCallStatefulPartitionedCallfeature_inputvalue_conv_37704607value_conv_37704609*
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
H__inference_value_conv_layer_call_and_return_conditional_losses_377045962$
"value_conv/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallfeature_inputconv2d_1_37704633conv2d_1_37704635*
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
GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_377046222"
 conv2d_1/StatefulPartitionedCall?
value_reshape/PartitionedCallPartitionedCall+value_conv/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *T
fORM
K__inference_value_reshape_layer_call_and_return_conditional_losses_377046502
value_reshape/PartitionedCall?
#value_dense/StatefulPartitionedCallStatefulPartitionedCall&value_reshape/PartitionedCall:output:0value_dense_37704679value_dense_37704681*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_value_dense_layer_call_and_return_conditional_losses_377046682%
#value_dense/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_377046962
reshape_1/PartitionedCall?
add_1/PartitionedCallPartitionedCall,value_dense/StatefulPartitionedCall:output:0"reshape_1/PartitionedCall:output:0*
Tin
2*
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
GPU 2J 8? *L
fGRE
C__inference_add_1_layer_call_and_return_conditional_losses_377047102
add_1/PartitionedCall?
lambda_1/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
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
F__inference_lambda_1_layer_call_and_return_conditional_losses_377047252
lambda_1/PartitionedCall?
IdentityIdentity!lambda_1/PartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall#^value_conv/StatefulPartitionedCall$^value_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2H
"value_conv/StatefulPartitionedCall"value_conv/StatefulPartitionedCall2J
#value_dense/StatefulPartitionedCall#value_dense/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????R
'
_user_specified_namefeature_input
?
o
C__inference_add_1_layer_call_and_return_conditional_losses_37705081
inputs_0
inputs_1
identityY
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
*__inference_model_1_layer_call_fn_37704849
feature_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeature_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_377048342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????R
'
_user_specified_namefeature_input
?	
?
H__inference_value_conv_layer_call_and_return_conditional_losses_37704994

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
?
$__inference__traced_restore_37705178
file_prefix&
"assignvariableop_value_conv_kernel&
"assignvariableop_1_value_conv_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias)
%assignvariableop_4_value_dense_kernel'
#assignvariableop_5_value_dense_bias

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp"assignvariableop_value_conv_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_value_conv_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_value_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_value_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6?

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_37705093

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ?E2
	truediv/yk
truedivRealDivinputstruediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truediv_
IdentityIdentitytruediv:z:0*
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
?
L
0__inference_value_reshape_layer_call_fn_37705020

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
GPU 2J 8? *T
fORM
K__inference_value_reshape_layer_call_and_return_conditional_losses_377046502
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
 
_user_specified_nameinputs
?
?
*__inference_model_1_layer_call_fn_37704967

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_377047942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?

g
K__inference_value_reshape_layer_call_and_return_conditional_losses_37705015

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
?
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_37704725

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ?E2
	truediv/yk
truedivRealDivinputstruediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truediv_
IdentityIdentitytruediv:z:0*
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
?
?
*__inference_model_1_layer_call_fn_37704809
feature_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeature_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_377047942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????R
'
_user_specified_namefeature_input
?	
?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_37704622

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
?5
?
E__inference_model_1_layer_call_and_return_conditional_losses_37704950

inputs-
)value_conv_conv2d_readvariableop_resource.
*value_conv_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource.
*value_dense_matmul_readvariableop_resource/
+value_dense_biasadd_readvariableop_resource
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?!value_conv/BiasAdd/ReadVariableOp? value_conv/Conv2D/ReadVariableOp?"value_dense/BiasAdd/ReadVariableOp?!value_dense/MatMul/ReadVariableOp?
 value_conv/Conv2D/ReadVariableOpReadVariableOp)value_conv_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02"
 value_conv/Conv2D/ReadVariableOp?
value_conv/Conv2DConv2Dinputs(value_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
value_conv/Conv2D?
!value_conv/BiasAdd/ReadVariableOpReadVariableOp*value_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!value_conv/BiasAdd/ReadVariableOp?
value_conv/BiasAddBiasAddvalue_conv/Conv2D:output:0)value_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
value_conv/BiasAdd?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_1/BiasAddu
value_reshape/ShapeShapevalue_conv/BiasAdd:output:0*
T0*
_output_shapes
:2
value_reshape/Shape?
!value_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!value_reshape/strided_slice/stack?
#value_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#value_reshape/strided_slice/stack_1?
#value_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#value_reshape/strided_slice/stack_2?
value_reshape/strided_sliceStridedSlicevalue_reshape/Shape:output:0*value_reshape/strided_slice/stack:output:0,value_reshape/strided_slice/stack_1:output:0,value_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
value_reshape/strided_slice?
value_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
value_reshape/Reshape/shape/1?
value_reshape/Reshape/shapePack$value_reshape/strided_slice:output:0&value_reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
value_reshape/Reshape/shape?
value_reshape/ReshapeReshapevalue_conv/BiasAdd:output:0$value_reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
value_reshape/Reshape?
!value_dense/MatMul/ReadVariableOpReadVariableOp*value_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!value_dense/MatMul/ReadVariableOp?
value_dense/MatMulMatMulvalue_reshape/Reshape:output:0)value_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
value_dense/MatMul?
"value_dense/BiasAdd/ReadVariableOpReadVariableOp+value_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"value_dense/BiasAdd/ReadVariableOp?
value_dense/BiasAddBiasAddvalue_dense/MatMul:product:0*value_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
value_dense/BiasAddk
reshape_1/ShapeShapeconv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slice?
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_1/Reshape/shape/1?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeconv2d_1/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
reshape_1/Reshape?
	add_1/addAddV2value_dense/BiasAdd:output:0reshape_1/Reshape:output:0*
T0*'
_output_shapes
:?????????2
	add_1/addm
lambda_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ?E2
lambda_1/truediv/y?
lambda_1/truedivRealDivadd_1/add:z:0lambda_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/truediv?
IdentityIdentitylambda_1/truediv:z:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp"^value_conv/BiasAdd/ReadVariableOp!^value_conv/Conv2D/ReadVariableOp#^value_dense/BiasAdd/ReadVariableOp"^value_dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R::::::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2F
!value_conv/BiasAdd/ReadVariableOp!value_conv/BiasAdd/ReadVariableOp2D
 value_conv/Conv2D/ReadVariableOp value_conv/Conv2D/ReadVariableOp2H
"value_dense/BiasAdd/ReadVariableOp"value_dense/BiasAdd/ReadVariableOp2F
!value_dense/MatMul/ReadVariableOp!value_dense/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_37705030

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
?
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_37705099

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ?E2
	truediv/yk
truedivRealDivinputstruediv/y:output:0*
T0*'
_output_shapes
:?????????2	
truediv_
IdentityIdentitytruediv:z:0*
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
?
G
+__inference_lambda_1_layer_call_fn_37705109

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
F__inference_lambda_1_layer_call_and_return_conditional_losses_377047312
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
?
?
!__inference__traced_save_37705150
file_prefix0
,savev2_value_conv_kernel_read_readvariableop.
*savev2_value_conv_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop1
-savev2_value_dense_kernel_read_readvariableop/
+savev2_value_dense_bias_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_value_conv_kernel_read_readvariableop*savev2_value_conv_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop-savev2_value_dense_kernel_read_readvariableop+savev2_value_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
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

identity_1Identity_1:output:0*W
_input_shapesF
D: :R::R:::: 2(
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
::,(
&
_output_shapes
:R: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?	
?
H__inference_value_conv_layer_call_and_return_conditional_losses_37704596

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
?5
?
E__inference_model_1_layer_call_and_return_conditional_losses_37704909

inputs-
)value_conv_conv2d_readvariableop_resource.
*value_conv_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource.
*value_dense_matmul_readvariableop_resource/
+value_dense_biasadd_readvariableop_resource
identity??conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?!value_conv/BiasAdd/ReadVariableOp? value_conv/Conv2D/ReadVariableOp?"value_dense/BiasAdd/ReadVariableOp?!value_dense/MatMul/ReadVariableOp?
 value_conv/Conv2D/ReadVariableOpReadVariableOp)value_conv_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02"
 value_conv/Conv2D/ReadVariableOp?
value_conv/Conv2DConv2Dinputs(value_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
value_conv/Conv2D?
!value_conv/BiasAdd/ReadVariableOpReadVariableOp*value_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!value_conv/BiasAdd/ReadVariableOp?
value_conv/BiasAddBiasAddvalue_conv/Conv2D:output:0)value_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
value_conv/BiasAdd?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_1/BiasAddu
value_reshape/ShapeShapevalue_conv/BiasAdd:output:0*
T0*
_output_shapes
:2
value_reshape/Shape?
!value_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!value_reshape/strided_slice/stack?
#value_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#value_reshape/strided_slice/stack_1?
#value_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#value_reshape/strided_slice/stack_2?
value_reshape/strided_sliceStridedSlicevalue_reshape/Shape:output:0*value_reshape/strided_slice/stack:output:0,value_reshape/strided_slice/stack_1:output:0,value_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
value_reshape/strided_slice?
value_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
value_reshape/Reshape/shape/1?
value_reshape/Reshape/shapePack$value_reshape/strided_slice:output:0&value_reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
value_reshape/Reshape/shape?
value_reshape/ReshapeReshapevalue_conv/BiasAdd:output:0$value_reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
value_reshape/Reshape?
!value_dense/MatMul/ReadVariableOpReadVariableOp*value_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!value_dense/MatMul/ReadVariableOp?
value_dense/MatMulMatMulvalue_reshape/Reshape:output:0)value_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
value_dense/MatMul?
"value_dense/BiasAdd/ReadVariableOpReadVariableOp+value_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"value_dense/BiasAdd/ReadVariableOp?
value_dense/BiasAddBiasAddvalue_dense/MatMul:product:0*value_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
value_dense/BiasAddk
reshape_1/ShapeShapeconv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slice?
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape_1/Reshape/shape/1?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeconv2d_1/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
reshape_1/Reshape?
	add_1/addAddV2value_dense/BiasAdd:output:0reshape_1/Reshape:output:0*
T0*'
_output_shapes
:?????????2
	add_1/addm
lambda_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ?E2
lambda_1/truediv/y?
lambda_1/truedivRealDivadd_1/add:z:0lambda_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/truediv?
IdentityIdentitylambda_1/truediv:z:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp"^value_conv/BiasAdd/ReadVariableOp!^value_conv/Conv2D/ReadVariableOp#^value_dense/BiasAdd/ReadVariableOp"^value_dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R::::::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2F
!value_conv/BiasAdd/ReadVariableOp!value_conv/BiasAdd/ReadVariableOp2D
 value_conv/Conv2D/ReadVariableOp value_conv/Conv2D/ReadVariableOp2H
"value_dense/BiasAdd/ReadVariableOp"value_dense/BiasAdd/ReadVariableOp2F
!value_dense/MatMul/ReadVariableOp!value_dense/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_37704868
feature_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeature_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_377045822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????R
'
_user_specified_namefeature_input
?
G
+__inference_lambda_1_layer_call_fn_37705104

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
F__inference_lambda_1_layer_call_and_return_conditional_losses_377047252
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
?
H
,__inference_reshape_1_layer_call_fn_37705075

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
GPU 2J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_377046962
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
 
_user_specified_nameinputs
?	
?
I__inference_value_dense_layer_call_and_return_conditional_losses_37704668

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
m
C__inference_add_1_layer_call_and_return_conditional_losses_37704710

inputs
inputs_1
identityW
addAddV2inputsinputs_1*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_1_layer_call_fn_37705039

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
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_377046222
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
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
O
feature_input>
serving_default_feature_input:0?????????R<
lambda_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer-7
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
X__call__
Y_default_save_signature
*Z&call_and_return_all_conditional_losses"?<
_tf_keras_network?<{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "feature_input"}, "name": "feature_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "value_conv", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value_conv", "inbound_nodes": [[["feature_input", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "value_reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}, "name": "value_reshape", "inbound_nodes": [[["value_conv", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["feature_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "value_dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value_dense", "inbound_nodes": [[["value_reshape", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}, "name": "reshape_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["value_dense", 0, 0, {}], ["reshape_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTmcAAAAAADijQKkAKQHaAXhyAQAA\nAHIBAAAA+kIvaG9tZS9zd2FuZy9wcm9qZWN0L1NtYXJ0V2Fsa2VyL1JMX1NpbXVsYXRpb24vY29k\nZS9hZ2VudF9tb2RlbHMucHnaCDxsYW1iZGE+RQAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "agent_models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["add_1", 0, 0, {}]]]}], "input_layers": [["feature_input", 0, 0]], "output_layers": [["lambda_1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 8, 82]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "feature_input"}, "name": "feature_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "value_conv", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value_conv", "inbound_nodes": [[["feature_input", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "value_reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}, "name": "value_reshape", "inbound_nodes": [[["value_conv", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["feature_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "value_dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value_dense", "inbound_nodes": [[["value_reshape", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}, "name": "reshape_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["value_dense", 0, 0, {}], ["reshape_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTmcAAAAAADijQKkAKQHaAXhyAQAA\nAHIBAAAA+kIvaG9tZS9zd2FuZy9wcm9qZWN0L1NtYXJ0V2Fsa2VyL1JMX1NpbXVsYXRpb24vY29k\nZS9hZ2VudF9tb2RlbHMucHnaCDxsYW1iZGE+RQAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "agent_models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda_1", "inbound_nodes": [[["add_1", 0, 0, {}]]]}], "input_layers": [["feature_input", 0, 0]], "output_layers": [["lambda_1", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "feature_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "feature_input"}}
?	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
[__call__
*\&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "value_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value_conv", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 82}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 8, 82]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "value_reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value_reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}}
?	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 82}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 8, 82]}}
?

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
a__call__
*b&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "value_dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value_dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
?
$regularization_losses
%	variables
&trainable_variables
'	keras_api
c__call__
*d&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}}
?
(regularization_losses
)	variables
*trainable_variables
+	keras_api
e__call__
*f&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 8]}]}
?
,regularization_losses
-	variables
.trainable_variables
/	keras_api
g__call__
*h&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTmcAAAAAADijQKkAKQHaAXhyAQAA\nAHIBAAAA+kIvaG9tZS9zd2FuZy9wcm9qZWN0L1NtYXJ0V2Fsa2VyL1JMX1NpbXVsYXRpb24vY29k\nZS9hZ2VudF9tb2RlbHMucHnaCDxsYW1iZGE+RQAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "agent_models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?

0layers
	regularization_losses

	variables
trainable_variables
1layer_regularization_losses
2layer_metrics
3metrics
4non_trainable_variables
X__call__
Y_default_save_signature
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
,
iserving_default"
signature_map
+:)R2value_conv/kernel
:2value_conv/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

5layers
regularization_losses
	variables
trainable_variables
6layer_regularization_losses
7layer_metrics
8metrics
9non_trainable_variables
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

:layers
regularization_losses
	variables
trainable_variables
;layer_regularization_losses
<layer_metrics
=metrics
>non_trainable_variables
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
):'R2conv2d_1/kernel
:2conv2d_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

?layers
regularization_losses
	variables
trainable_variables
@layer_regularization_losses
Alayer_metrics
Bmetrics
Cnon_trainable_variables
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
$:"2value_dense/kernel
:2value_dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Dlayers
 regularization_losses
!	variables
"trainable_variables
Elayer_regularization_losses
Flayer_metrics
Gmetrics
Hnon_trainable_variables
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Ilayers
$regularization_losses
%	variables
&trainable_variables
Jlayer_regularization_losses
Klayer_metrics
Lmetrics
Mnon_trainable_variables
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Nlayers
(regularization_losses
)	variables
*trainable_variables
Olayer_regularization_losses
Player_metrics
Qmetrics
Rnon_trainable_variables
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Slayers
,regularization_losses
-	variables
.trainable_variables
Tlayer_regularization_losses
Ulayer_metrics
Vmetrics
Wnon_trainable_variables
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
X
0
1
2
3
4
5
6
7"
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
?2?
*__inference_model_1_layer_call_fn_37704967
*__inference_model_1_layer_call_fn_37704849
*__inference_model_1_layer_call_fn_37704984
*__inference_model_1_layer_call_fn_37704809?
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
#__inference__wrapped_model_37704582?
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
annotations? *4?1
/?,
feature_input?????????R
?2?
E__inference_model_1_layer_call_and_return_conditional_losses_37704950
E__inference_model_1_layer_call_and_return_conditional_losses_37704745
E__inference_model_1_layer_call_and_return_conditional_losses_37704768
E__inference_model_1_layer_call_and_return_conditional_losses_37704909?
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
-__inference_value_conv_layer_call_fn_37705003?
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
H__inference_value_conv_layer_call_and_return_conditional_losses_37704994?
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
0__inference_value_reshape_layer_call_fn_37705020?
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
K__inference_value_reshape_layer_call_and_return_conditional_losses_37705015?
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
+__inference_conv2d_1_layer_call_fn_37705039?
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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_37705030?
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
.__inference_value_dense_layer_call_fn_37705058?
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
I__inference_value_dense_layer_call_and_return_conditional_losses_37705049?
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
,__inference_reshape_1_layer_call_fn_37705075?
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
G__inference_reshape_1_layer_call_and_return_conditional_losses_37705070?
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
(__inference_add_1_layer_call_fn_37705087?
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
C__inference_add_1_layer_call_and_return_conditional_losses_37705081?
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
?2?
+__inference_lambda_1_layer_call_fn_37705109
+__inference_lambda_1_layer_call_fn_37705104?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_lambda_1_layer_call_and_return_conditional_losses_37705093
F__inference_lambda_1_layer_call_and_return_conditional_losses_37705099?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
&__inference_signature_wrapper_37704868feature_input"?
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
#__inference__wrapped_model_37704582}>?;
4?1
/?,
feature_input?????????R
? "3?0
.
lambda_1"?
lambda_1??????????
C__inference_add_1_layer_call_and_return_conditional_losses_37705081?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
(__inference_add_1_layer_call_fn_37705087vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
F__inference_conv2d_1_layer_call_and_return_conditional_losses_37705030l7?4
-?*
(?%
inputs?????????R
? "-?*
#? 
0?????????
? ?
+__inference_conv2d_1_layer_call_fn_37705039_7?4
-?*
(?%
inputs?????????R
? " ???????????
F__inference_lambda_1_layer_call_and_return_conditional_losses_37705093`7?4
-?*
 ?
inputs?????????

 
p
? "%?"
?
0?????????
? ?
F__inference_lambda_1_layer_call_and_return_conditional_losses_37705099`7?4
-?*
 ?
inputs?????????

 
p 
? "%?"
?
0?????????
? ?
+__inference_lambda_1_layer_call_fn_37705104S7?4
-?*
 ?
inputs?????????

 
p
? "???????????
+__inference_lambda_1_layer_call_fn_37705109S7?4
-?*
 ?
inputs?????????

 
p 
? "???????????
E__inference_model_1_layer_call_and_return_conditional_losses_37704745wF?C
<?9
/?,
feature_input?????????R
p

 
? "%?"
?
0?????????
? ?
E__inference_model_1_layer_call_and_return_conditional_losses_37704768wF?C
<?9
/?,
feature_input?????????R
p 

 
? "%?"
?
0?????????
? ?
E__inference_model_1_layer_call_and_return_conditional_losses_37704909p??<
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
E__inference_model_1_layer_call_and_return_conditional_losses_37704950p??<
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
*__inference_model_1_layer_call_fn_37704809jF?C
<?9
/?,
feature_input?????????R
p

 
? "???????????
*__inference_model_1_layer_call_fn_37704849jF?C
<?9
/?,
feature_input?????????R
p 

 
? "???????????
*__inference_model_1_layer_call_fn_37704967c??<
5?2
(?%
inputs?????????R
p

 
? "???????????
*__inference_model_1_layer_call_fn_37704984c??<
5?2
(?%
inputs?????????R
p 

 
? "???????????
G__inference_reshape_1_layer_call_and_return_conditional_losses_37705070`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????
? ?
,__inference_reshape_1_layer_call_fn_37705075S7?4
-?*
(?%
inputs?????????
? "???????????
&__inference_signature_wrapper_37704868?O?L
? 
E?B
@
feature_input/?,
feature_input?????????R"3?0
.
lambda_1"?
lambda_1??????????
H__inference_value_conv_layer_call_and_return_conditional_losses_37704994l7?4
-?*
(?%
inputs?????????R
? "-?*
#? 
0?????????
? ?
-__inference_value_conv_layer_call_fn_37705003_7?4
-?*
(?%
inputs?????????R
? " ???????????
I__inference_value_dense_layer_call_and_return_conditional_losses_37705049\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
.__inference_value_dense_layer_call_fn_37705058O/?,
%?"
 ?
inputs?????????
? "???????????
K__inference_value_reshape_layer_call_and_return_conditional_losses_37705015`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????
? ?
0__inference_value_reshape_layer_call_fn_37705020S7?4
-?*
(?%
inputs?????????
? "??????????