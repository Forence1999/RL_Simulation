¼ 
±
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8

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
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:R*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:R*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0

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
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/value_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:R*)
shared_nameAdam/value_conv/kernel/m

,Adam/value_conv/kernel/m/Read/ReadVariableOpReadVariableOpAdam/value_conv/kernel/m*&
_output_shapes
:R*
dtype0

Adam/value_conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/value_conv/bias/m
}
*Adam/value_conv/bias/m/Read/ReadVariableOpReadVariableOpAdam/value_conv/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:R*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:R*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0

Adam/value_dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameAdam/value_dense/kernel/m

-Adam/value_dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/value_dense/kernel/m*
_output_shapes

:*
dtype0

Adam/value_dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/value_dense/bias/m

+Adam/value_dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/value_dense/bias/m*
_output_shapes
:*
dtype0

Adam/value_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:R*)
shared_nameAdam/value_conv/kernel/v

,Adam/value_conv/kernel/v/Read/ReadVariableOpReadVariableOpAdam/value_conv/kernel/v*&
_output_shapes
:R*
dtype0

Adam/value_conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/value_conv/bias/v
}
*Adam/value_conv/bias/v/Read/ReadVariableOpReadVariableOpAdam/value_conv/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:R*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:R*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0

Adam/value_dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameAdam/value_dense/kernel/v

-Adam/value_dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/value_dense/kernel/v*
_output_shapes

:*
dtype0

Adam/value_dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/value_dense/bias/v

+Adam/value_dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/value_dense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
è.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*£.
value.B. B.
§
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
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
 bias
!	variables
"regularization_losses
#trainable_variables
$	keras_api
R
%	variables
&regularization_losses
'trainable_variables
(	keras_api
R
)	variables
*regularization_losses
+trainable_variables
,	keras_api
R
-	variables
.regularization_losses
/trainable_variables
0	keras_api
¬
1iter

2beta_1

3beta_2
	4decay
5learning_ratemimjmkmlmm mnvovpvqvrvs vt
 
*
0
1
2
3
4
 5
*
0
1
2
3
4
 5
­

6layers
7layer_regularization_losses
8metrics
9non_trainable_variables

regularization_losses
:layer_metrics
trainable_variables
	variables
 
][
VARIABLE_VALUEvalue_conv/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEvalue_conv/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

;layers
<layer_regularization_losses
=metrics
>non_trainable_variables
	variables
regularization_losses
?layer_metrics
trainable_variables
 
 
 
­

@layers
Alayer_regularization_losses
Bmetrics
Cnon_trainable_variables
	variables
regularization_losses
Dlayer_metrics
trainable_variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

Elayers
Flayer_regularization_losses
Gmetrics
Hnon_trainable_variables
	variables
regularization_losses
Ilayer_metrics
trainable_variables
^\
VARIABLE_VALUEvalue_dense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEvalue_dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1
 

0
 1
­

Jlayers
Klayer_regularization_losses
Lmetrics
Mnon_trainable_variables
!	variables
"regularization_losses
Nlayer_metrics
#trainable_variables
 
 
 
­

Olayers
Player_regularization_losses
Qmetrics
Rnon_trainable_variables
%	variables
&regularization_losses
Slayer_metrics
'trainable_variables
 
 
 
­

Tlayers
Ulayer_regularization_losses
Vmetrics
Wnon_trainable_variables
)	variables
*regularization_losses
Xlayer_metrics
+trainable_variables
 
 
 
­

Ylayers
Zlayer_regularization_losses
[metrics
\non_trainable_variables
-	variables
.regularization_losses
]layer_metrics
/trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
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

^0
_1
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
4
	`total
	acount
b	variables
c	keras_api
D
	dtotal
	ecount
f
_fn_kwargs
g	variables
h	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

`0
a1

b	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

d0
e1

g	variables
~
VARIABLE_VALUEAdam/value_conv/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/value_conv/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/value_dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/value_dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/value_conv/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/value_conv/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/value_dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/value_dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_feature_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿR
¬
StatefulPartitionedCallStatefulPartitionedCallserving_default_feature_inputvalue_conv/kernelvalue_conv/biasconv2d/kernelconv2d/biasvalue_dense/kernelvalue_dense/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_36670206
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¿

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%value_conv/kernel/Read/ReadVariableOp#value_conv/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp&value_dense/kernel/Read/ReadVariableOp$value_dense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/value_conv/kernel/m/Read/ReadVariableOp*Adam/value_conv/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp-Adam/value_dense/kernel/m/Read/ReadVariableOp+Adam/value_dense/bias/m/Read/ReadVariableOp,Adam/value_conv/kernel/v/Read/ReadVariableOp*Adam/value_conv/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp-Adam/value_dense/kernel/v/Read/ReadVariableOp+Adam/value_dense/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
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
GPU 2J 8 **
f%R#
!__inference__traced_save_36670551

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamevalue_conv/kernelvalue_conv/biasconv2d/kernelconv2d/biasvalue_dense/kernelvalue_dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/value_conv/kernel/mAdam/value_conv/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/value_dense/kernel/mAdam/value_dense/bias/mAdam/value_conv/kernel/vAdam/value_conv/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/value_dense/kernel/vAdam/value_dense/bias/v*'
Tin 
2*
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_36670642£
°
L
0__inference_value_reshape_layer_call_fn_36670358

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_value_reshape_layer_call_and_return_conditional_losses_366699802
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
F
*__inference_reshape_layer_call_fn_36670413

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_366700262
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï

a
E__inference_reshape_layer_call_and_return_conditional_losses_36670026

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
strided_slice/stack_2â
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
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/1
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

å
C__inference_model_layer_call_and_return_conditional_losses_36670075
feature_input
value_conv_36669937
value_conv_36669939
conv2d_36669963
conv2d_36669965
value_dense_36670009
value_dense_36670011
identity¢conv2d/StatefulPartitionedCall¢"value_conv/StatefulPartitionedCall¢#value_dense/StatefulPartitionedCall³
"value_conv/StatefulPartitionedCallStatefulPartitionedCallfeature_inputvalue_conv_36669937value_conv_36669939*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_value_conv_layer_call_and_return_conditional_losses_366699262$
"value_conv/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallfeature_inputconv2d_36669963conv2d_36669965*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_366699522 
conv2d/StatefulPartitionedCall
value_reshape/PartitionedCallPartitionedCall+value_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_value_reshape_layer_call_and_return_conditional_losses_366699802
value_reshape/PartitionedCallÉ
#value_dense/StatefulPartitionedCallStatefulPartitionedCall&value_reshape/PartitionedCall:output:0value_dense_36670009value_dense_36670011*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_value_dense_layer_call_and_return_conditional_losses_366699982%
#value_dense/StatefulPartitionedCallô
reshape/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_366700262
reshape/PartitionedCall
add/PartitionedCallPartitionedCall,value_dense/StatefulPartitionedCall:output:0 reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_add_layer_call_and_return_conditional_losses_366700402
add/PartitionedCallæ
lambda/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_366700552
lambda/PartitionedCallß
IdentityIdentitylambda/PartitionedCall:output:0^conv2d/StatefulPartitionedCall#^value_conv/StatefulPartitionedCall$^value_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿR::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2H
"value_conv/StatefulPartitionedCall"value_conv/StatefulPartitionedCall2J
#value_dense/StatefulPartitionedCall#value_dense/StatefulPartitionedCall:^ Z
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
'
_user_specified_namefeature_input
ï	
á
H__inference_value_conv_layer_call_and_return_conditional_losses_36670332

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿR::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs
4
È
C__inference_model_layer_call_and_return_conditional_losses_36670288

inputs-
)value_conv_conv2d_readvariableop_resource.
*value_conv_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource.
*value_dense_matmul_readvariableop_resource/
+value_dense_biasadd_readvariableop_resource
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢!value_conv/BiasAdd/ReadVariableOp¢ value_conv/Conv2D/ReadVariableOp¢"value_dense/BiasAdd/ReadVariableOp¢!value_dense/MatMul/ReadVariableOp¶
 value_conv/Conv2D/ReadVariableOpReadVariableOp)value_conv_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02"
 value_conv/Conv2D/ReadVariableOpÅ
value_conv/Conv2DConv2Dinputs(value_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
value_conv/Conv2D­
!value_conv/BiasAdd/ReadVariableOpReadVariableOp*value_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!value_conv/BiasAdd/ReadVariableOp´
value_conv/BiasAddBiasAddvalue_conv/Conv2D:output:0)value_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
value_conv/BiasAddª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02
conv2d/Conv2D/ReadVariableOp¹
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/BiasAddu
value_reshape/ShapeShapevalue_conv/BiasAdd:output:0*
T0*
_output_shapes
:2
value_reshape/Shape
!value_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!value_reshape/strided_slice/stack
#value_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#value_reshape/strided_slice/stack_1
#value_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#value_reshape/strided_slice/stack_2¶
value_reshape/strided_sliceStridedSlicevalue_reshape/Shape:output:0*value_reshape/strided_slice/stack:output:0,value_reshape/strided_slice/stack_1:output:0,value_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
value_reshape/strided_slice
value_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
value_reshape/Reshape/shape/1¾
value_reshape/Reshape/shapePack$value_reshape/strided_slice:output:0&value_reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
value_reshape/Reshape/shape®
value_reshape/ReshapeReshapevalue_conv/BiasAdd:output:0$value_reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
value_reshape/Reshape±
!value_dense/MatMul/ReadVariableOpReadVariableOp*value_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!value_dense/MatMul/ReadVariableOp¯
value_dense/MatMulMatMulvalue_reshape/Reshape:output:0)value_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
value_dense/MatMul°
"value_dense/BiasAdd/ReadVariableOpReadVariableOp+value_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"value_dense/BiasAdd/ReadVariableOp±
value_dense/BiasAddBiasAddvalue_dense/MatMul:product:0*value_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
value_dense/BiasAdde
reshape/ShapeShapeconv2d/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
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
ÿÿÿÿÿÿÿÿÿ2
reshape/Reshape/shape/1¦
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape
reshape/ReshapeReshapeconv2d/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reshape/Reshape
add/addAddV2value_dense/BiasAdd:output:0reshape/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
add/addi
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ÀE2
lambda/truediv/y
lambda/truedivRealDivadd/add:z:0lambda/truediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lambda/truedivµ
IdentityIdentitylambda/truediv:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp"^value_conv/BiasAdd/ReadVariableOp!^value_conv/Conv2D/ReadVariableOp#^value_dense/BiasAdd/ReadVariableOp"^value_dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿR::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2F
!value_conv/BiasAdd/ReadVariableOp!value_conv/BiasAdd/ReadVariableOp2D
 value_conv/Conv2D/ReadVariableOp value_conv/Conv2D/ReadVariableOp2H
"value_dense/BiasAdd/ReadVariableOp"value_dense/BiasAdd/ReadVariableOp2F
!value_dense/MatMul/ReadVariableOp!value_dense/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs
	
â
I__inference_value_dense_layer_call_and_return_conditional_losses_36669998

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
m
A__inference_add_layer_call_and_return_conditional_losses_36670419
inputs_0
inputs_1
identityY
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ï

a
E__inference_reshape_layer_call_and_return_conditional_losses_36670408

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
strided_slice/stack_2â
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
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/1
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
4
È
C__inference_model_layer_call_and_return_conditional_losses_36670247

inputs-
)value_conv_conv2d_readvariableop_resource.
*value_conv_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource.
*value_dense_matmul_readvariableop_resource/
+value_dense_biasadd_readvariableop_resource
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢!value_conv/BiasAdd/ReadVariableOp¢ value_conv/Conv2D/ReadVariableOp¢"value_dense/BiasAdd/ReadVariableOp¢!value_dense/MatMul/ReadVariableOp¶
 value_conv/Conv2D/ReadVariableOpReadVariableOp)value_conv_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02"
 value_conv/Conv2D/ReadVariableOpÅ
value_conv/Conv2DConv2Dinputs(value_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
value_conv/Conv2D­
!value_conv/BiasAdd/ReadVariableOpReadVariableOp*value_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!value_conv/BiasAdd/ReadVariableOp´
value_conv/BiasAddBiasAddvalue_conv/Conv2D:output:0)value_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
value_conv/BiasAddª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02
conv2d/Conv2D/ReadVariableOp¹
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d/BiasAddu
value_reshape/ShapeShapevalue_conv/BiasAdd:output:0*
T0*
_output_shapes
:2
value_reshape/Shape
!value_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!value_reshape/strided_slice/stack
#value_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#value_reshape/strided_slice/stack_1
#value_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#value_reshape/strided_slice/stack_2¶
value_reshape/strided_sliceStridedSlicevalue_reshape/Shape:output:0*value_reshape/strided_slice/stack:output:0,value_reshape/strided_slice/stack_1:output:0,value_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
value_reshape/strided_slice
value_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
value_reshape/Reshape/shape/1¾
value_reshape/Reshape/shapePack$value_reshape/strided_slice:output:0&value_reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
value_reshape/Reshape/shape®
value_reshape/ReshapeReshapevalue_conv/BiasAdd:output:0$value_reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
value_reshape/Reshape±
!value_dense/MatMul/ReadVariableOpReadVariableOp*value_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!value_dense/MatMul/ReadVariableOp¯
value_dense/MatMulMatMulvalue_reshape/Reshape:output:0)value_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
value_dense/MatMul°
"value_dense/BiasAdd/ReadVariableOpReadVariableOp+value_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"value_dense/BiasAdd/ReadVariableOp±
value_dense/BiasAddBiasAddvalue_dense/MatMul:product:0*value_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
value_dense/BiasAdde
reshape/ShapeShapeconv2d/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
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
ÿÿÿÿÿÿÿÿÿ2
reshape/Reshape/shape/1¦
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape
reshape/ReshapeReshapeconv2d/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reshape/Reshape
add/addAddV2value_dense/BiasAdd:output:0reshape/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
add/addi
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ÀE2
lambda/truediv/y
lambda/truedivRealDivadd/add:z:0lambda/truediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lambda/truedivµ
IdentityIdentitylambda/truediv:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp"^value_conv/BiasAdd/ReadVariableOp!^value_conv/Conv2D/ReadVariableOp#^value_dense/BiasAdd/ReadVariableOp"^value_dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿR::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2F
!value_conv/BiasAdd/ReadVariableOp!value_conv/BiasAdd/ReadVariableOp2D
 value_conv/Conv2D/ReadVariableOp value_conv/Conv2D/ReadVariableOp2H
"value_dense/BiasAdd/ReadVariableOp"value_dense/BiasAdd/ReadVariableOp2F
!value_dense/MatMul/ReadVariableOp!value_dense/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs
¥
`
D__inference_lambda_layer_call_and_return_conditional_losses_36670431

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ÀE2
	truediv/yk
truedivRealDivinputstruediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
â
I__inference_value_dense_layer_call_and_return_conditional_losses_36670387

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ

g
K__inference_value_reshape_layer_call_and_return_conditional_losses_36669980

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
strided_slice/stack_2â
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
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/1
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
`
D__inference_lambda_layer_call_and_return_conditional_losses_36670061

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ÀE2
	truediv/yk
truedivRealDivinputstruediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


-__inference_value_conv_layer_call_fn_36670341

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_value_conv_layer_call_and_return_conditional_losses_366699262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿR::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs

À
(__inference_model_layer_call_fn_36670179
feature_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallfeature_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_366701642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿR::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
'
_user_specified_namefeature_input

å
C__inference_model_layer_call_and_return_conditional_losses_36670098
feature_input
value_conv_36670078
value_conv_36670080
conv2d_36670083
conv2d_36670085
value_dense_36670089
value_dense_36670091
identity¢conv2d/StatefulPartitionedCall¢"value_conv/StatefulPartitionedCall¢#value_dense/StatefulPartitionedCall³
"value_conv/StatefulPartitionedCallStatefulPartitionedCallfeature_inputvalue_conv_36670078value_conv_36670080*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_value_conv_layer_call_and_return_conditional_losses_366699262$
"value_conv/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallfeature_inputconv2d_36670083conv2d_36670085*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_366699522 
conv2d/StatefulPartitionedCall
value_reshape/PartitionedCallPartitionedCall+value_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_value_reshape_layer_call_and_return_conditional_losses_366699802
value_reshape/PartitionedCallÉ
#value_dense/StatefulPartitionedCallStatefulPartitionedCall&value_reshape/PartitionedCall:output:0value_dense_36670089value_dense_36670091*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_value_dense_layer_call_and_return_conditional_losses_366699982%
#value_dense/StatefulPartitionedCallô
reshape/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_366700262
reshape/PartitionedCall
add/PartitionedCallPartitionedCall,value_dense/StatefulPartitionedCall:output:0 reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_add_layer_call_and_return_conditional_losses_366700402
add/PartitionedCallæ
lambda/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_366700612
lambda/PartitionedCallß
IdentityIdentitylambda/PartitionedCall:output:0^conv2d/StatefulPartitionedCall#^value_conv/StatefulPartitionedCall$^value_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿR::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2H
"value_conv/StatefulPartitionedCall"value_conv/StatefulPartitionedCall2J
#value_dense/StatefulPartitionedCall#value_dense/StatefulPartitionedCall:^ Z
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
'
_user_specified_namefeature_input
Þ
¾
&__inference_signature_wrapper_36670206
feature_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallfeature_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_366699122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿR::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
'
_user_specified_namefeature_input
¥
`
D__inference_lambda_layer_call_and_return_conditional_losses_36670055

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ÀE2
	truediv/yk
truedivRealDivinputstruediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë	
Ý
D__inference_conv2d_layer_call_and_return_conditional_losses_36669952

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿR::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs
ç

.__inference_value_dense_layer_call_fn_36670396

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_value_dense_layer_call_and_return_conditional_losses_366699982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
~
)__inference_conv2d_layer_call_fn_36670377

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_366699522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿR::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs

R
&__inference_add_layer_call_fn_36670425
inputs_0
inputs_1
identityÌ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_add_layer_call_and_return_conditional_losses_366700402
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ë
¹
(__inference_model_layer_call_fn_36670322

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_366701642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿR::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs
ë
¹
(__inference_model_layer_call_fn_36670305

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_366701242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿR::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs
ñ
Þ
C__inference_model_layer_call_and_return_conditional_losses_36670124

inputs
value_conv_36670104
value_conv_36670106
conv2d_36670109
conv2d_36670111
value_dense_36670115
value_dense_36670117
identity¢conv2d/StatefulPartitionedCall¢"value_conv/StatefulPartitionedCall¢#value_dense/StatefulPartitionedCall¬
"value_conv/StatefulPartitionedCallStatefulPartitionedCallinputsvalue_conv_36670104value_conv_36670106*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_value_conv_layer_call_and_return_conditional_losses_366699262$
"value_conv/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_36670109conv2d_36670111*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_366699522 
conv2d/StatefulPartitionedCall
value_reshape/PartitionedCallPartitionedCall+value_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_value_reshape_layer_call_and_return_conditional_losses_366699802
value_reshape/PartitionedCallÉ
#value_dense/StatefulPartitionedCallStatefulPartitionedCall&value_reshape/PartitionedCall:output:0value_dense_36670115value_dense_36670117*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_value_dense_layer_call_and_return_conditional_losses_366699982%
#value_dense/StatefulPartitionedCallô
reshape/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_366700262
reshape/PartitionedCall
add/PartitionedCallPartitionedCall,value_dense/StatefulPartitionedCall:output:0 reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_add_layer_call_and_return_conditional_losses_366700402
add/PartitionedCallæ
lambda/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_366700552
lambda/PartitionedCallß
IdentityIdentitylambda/PartitionedCall:output:0^conv2d/StatefulPartitionedCall#^value_conv/StatefulPartitionedCall$^value_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿR::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2H
"value_conv/StatefulPartitionedCall"value_conv/StatefulPartitionedCall2J
#value_dense/StatefulPartitionedCall#value_dense/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs

E
)__inference_lambda_layer_call_fn_36670447

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_366700612
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
`
D__inference_lambda_layer_call_and_return_conditional_losses_36670437

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ÀE2
	truediv/yk
truedivRealDivinputstruediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï	
á
H__inference_value_conv_layer_call_and_return_conditional_losses_36669926

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿR::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs
s
ø
$__inference__traced_restore_36670642
file_prefix&
"assignvariableop_value_conv_kernel&
"assignvariableop_1_value_conv_bias$
 assignvariableop_2_conv2d_kernel"
assignvariableop_3_conv2d_bias)
%assignvariableop_4_value_dense_kernel'
#assignvariableop_5_value_dense_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_10
,assignvariableop_15_adam_value_conv_kernel_m.
*assignvariableop_16_adam_value_conv_bias_m,
(assignvariableop_17_adam_conv2d_kernel_m*
&assignvariableop_18_adam_conv2d_bias_m1
-assignvariableop_19_adam_value_dense_kernel_m/
+assignvariableop_20_adam_value_dense_bias_m0
,assignvariableop_21_adam_value_conv_kernel_v.
*assignvariableop_22_adam_value_conv_bias_v,
(assignvariableop_23_adam_conv2d_kernel_v*
&assignvariableop_24_adam_conv2d_bias_v1
-assignvariableop_25_adam_value_dense_kernel_v/
+assignvariableop_26_adam_value_dense_bias_v
identity_28¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÆ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¸
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¡
AssignVariableOpAssignVariableOp"assignvariableop_value_conv_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_value_conv_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¥
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2d_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ª
AssignVariableOp_4AssignVariableOp%assignvariableop_4_value_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¨
AssignVariableOp_5AssignVariableOp#assignvariableop_5_value_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6¡
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¢
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¡
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¡
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13£
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14£
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15´
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_value_conv_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16²
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_value_conv_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17°
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_conv2d_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18®
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_conv2d_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19µ
AssignVariableOp_19AssignVariableOp-assignvariableop_19_adam_value_dense_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20³
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_value_dense_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21´
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_value_conv_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22²
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_value_conv_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23°
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_conv2d_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24®
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_conv2d_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25µ
AssignVariableOp_25AssignVariableOp-assignvariableop_25_adam_value_dense_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26³
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_value_dense_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp°
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27£
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*
_input_shapesp
n: :::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ñ
Þ
C__inference_model_layer_call_and_return_conditional_losses_36670164

inputs
value_conv_36670144
value_conv_36670146
conv2d_36670149
conv2d_36670151
value_dense_36670155
value_dense_36670157
identity¢conv2d/StatefulPartitionedCall¢"value_conv/StatefulPartitionedCall¢#value_dense/StatefulPartitionedCall¬
"value_conv/StatefulPartitionedCallStatefulPartitionedCallinputsvalue_conv_36670144value_conv_36670146*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_value_conv_layer_call_and_return_conditional_losses_366699262$
"value_conv/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_36670149conv2d_36670151*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_366699522 
conv2d/StatefulPartitionedCall
value_reshape/PartitionedCallPartitionedCall+value_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_value_reshape_layer_call_and_return_conditional_losses_366699802
value_reshape/PartitionedCallÉ
#value_dense/StatefulPartitionedCallStatefulPartitionedCall&value_reshape/PartitionedCall:output:0value_dense_36670155value_dense_36670157*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_value_dense_layer_call_and_return_conditional_losses_366699982%
#value_dense/StatefulPartitionedCallô
reshape/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_366700262
reshape/PartitionedCall
add/PartitionedCallPartitionedCall,value_dense/StatefulPartitionedCall:output:0 reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_add_layer_call_and_return_conditional_losses_366700402
add/PartitionedCallæ
lambda/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_366700612
lambda/PartitionedCallß
IdentityIdentitylambda/PartitionedCall:output:0^conv2d/StatefulPartitionedCall#^value_conv/StatefulPartitionedCall$^value_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿR::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2H
"value_conv/StatefulPartitionedCall"value_conv/StatefulPartitionedCall2J
#value_dense/StatefulPartitionedCall#value_dense/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs
ë	
Ý
D__inference_conv2d_layer_call_and_return_conditional_losses_36670368

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿR::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs
 :
÷
#__inference__wrapped_model_36669912
feature_input3
/model_value_conv_conv2d_readvariableop_resource4
0model_value_conv_biasadd_readvariableop_resource/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource4
0model_value_dense_matmul_readvariableop_resource5
1model_value_dense_biasadd_readvariableop_resource
identity¢#model/conv2d/BiasAdd/ReadVariableOp¢"model/conv2d/Conv2D/ReadVariableOp¢'model/value_conv/BiasAdd/ReadVariableOp¢&model/value_conv/Conv2D/ReadVariableOp¢(model/value_dense/BiasAdd/ReadVariableOp¢'model/value_dense/MatMul/ReadVariableOpÈ
&model/value_conv/Conv2D/ReadVariableOpReadVariableOp/model_value_conv_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02(
&model/value_conv/Conv2D/ReadVariableOpÞ
model/value_conv/Conv2DConv2Dfeature_input.model/value_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
model/value_conv/Conv2D¿
'model/value_conv/BiasAdd/ReadVariableOpReadVariableOp0model_value_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model/value_conv/BiasAdd/ReadVariableOpÌ
model/value_conv/BiasAddBiasAdd model/value_conv/Conv2D:output:0/model/value_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/value_conv/BiasAdd¼
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02$
"model/conv2d/Conv2D/ReadVariableOpÒ
model/conv2d/Conv2DConv2Dfeature_input*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
model/conv2d/Conv2D³
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp¼
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/conv2d/BiasAdd
model/value_reshape/ShapeShape!model/value_conv/BiasAdd:output:0*
T0*
_output_shapes
:2
model/value_reshape/Shape
'model/value_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model/value_reshape/strided_slice/stack 
)model/value_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model/value_reshape/strided_slice/stack_1 
)model/value_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model/value_reshape/strided_slice/stack_2Ú
!model/value_reshape/strided_sliceStridedSlice"model/value_reshape/Shape:output:00model/value_reshape/strided_slice/stack:output:02model/value_reshape/strided_slice/stack_1:output:02model/value_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!model/value_reshape/strided_slice
#model/value_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#model/value_reshape/Reshape/shape/1Ö
!model/value_reshape/Reshape/shapePack*model/value_reshape/strided_slice:output:0,model/value_reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2#
!model/value_reshape/Reshape/shapeÆ
model/value_reshape/ReshapeReshape!model/value_conv/BiasAdd:output:0*model/value_reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/value_reshape/ReshapeÃ
'model/value_dense/MatMul/ReadVariableOpReadVariableOp0model_value_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'model/value_dense/MatMul/ReadVariableOpÇ
model/value_dense/MatMulMatMul$model/value_reshape/Reshape:output:0/model/value_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/value_dense/MatMulÂ
(model/value_dense/BiasAdd/ReadVariableOpReadVariableOp1model_value_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model/value_dense/BiasAdd/ReadVariableOpÉ
model/value_dense/BiasAddBiasAdd"model/value_dense/MatMul:product:00model/value_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/value_dense/BiasAddw
model/reshape/ShapeShapemodel/conv2d/BiasAdd:output:0*
T0*
_output_shapes
:2
model/reshape/Shape
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model/reshape/strided_slice/stack
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#model/reshape/strided_slice/stack_1
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#model/reshape/strided_slice/stack_2¶
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/reshape/strided_slice
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
model/reshape/Reshape/shape/1¾
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
model/reshape/Reshape/shape°
model/reshape/ReshapeReshapemodel/conv2d/BiasAdd:output:0$model/reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/reshape/Reshape
model/add/addAddV2"model/value_dense/BiasAdd:output:0model/reshape/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/add/addu
model/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 * ÀE2
model/lambda/truediv/y
model/lambda/truedivRealDivmodel/add/add:z:0model/lambda/truediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/lambda/truedivß
IdentityIdentitymodel/lambda/truediv:z:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp(^model/value_conv/BiasAdd/ReadVariableOp'^model/value_conv/Conv2D/ReadVariableOp)^model/value_dense/BiasAdd/ReadVariableOp(^model/value_dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿR::::::2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2R
'model/value_conv/BiasAdd/ReadVariableOp'model/value_conv/BiasAdd/ReadVariableOp2P
&model/value_conv/Conv2D/ReadVariableOp&model/value_conv/Conv2D/ReadVariableOp2T
(model/value_dense/BiasAdd/ReadVariableOp(model/value_dense/BiasAdd/ReadVariableOp2R
'model/value_dense/MatMul/ReadVariableOp'model/value_dense/MatMul/ReadVariableOp:^ Z
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
'
_user_specified_namefeature_input

À
(__inference_model_layer_call_fn_36670139
feature_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallfeature_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_366701242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿR::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
'
_user_specified_namefeature_input
õ

g
K__inference_value_reshape_layer_call_and_return_conditional_losses_36670353

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
strided_slice/stack_2â
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
ÿÿÿÿÿÿÿÿÿ2
Reshape/shape/1
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦>

!__inference__traced_save_36670551
file_prefix0
,savev2_value_conv_kernel_read_readvariableop.
*savev2_value_conv_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop1
-savev2_value_dense_kernel_read_readvariableop/
+savev2_value_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_value_conv_kernel_m_read_readvariableop5
1savev2_adam_value_conv_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop8
4savev2_adam_value_dense_kernel_m_read_readvariableop6
2savev2_adam_value_dense_bias_m_read_readvariableop7
3savev2_adam_value_conv_kernel_v_read_readvariableop5
1savev2_adam_value_conv_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop8
4savev2_adam_value_dense_kernel_v_read_readvariableop6
2savev2_adam_value_dense_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÀ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_value_conv_kernel_read_readvariableop*savev2_value_conv_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop-savev2_value_dense_kernel_read_readvariableop+savev2_value_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_value_conv_kernel_m_read_readvariableop1savev2_adam_value_conv_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop4savev2_adam_value_dense_kernel_m_read_readvariableop2savev2_adam_value_dense_bias_m_read_readvariableop3savev2_adam_value_conv_kernel_v_read_readvariableop1savev2_adam_value_conv_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop4savev2_adam_value_dense_kernel_v_read_readvariableop2savev2_adam_value_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*ë
_input_shapesÙ
Ö: :R::R:::: : : : : : : : : :R::R::::R::R:::: 2(
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
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:R: 

_output_shapes
::,(
&
_output_shapes
:R: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:R: 

_output_shapes
::,(
&
_output_shapes
:R: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 

k
A__inference_add_layer_call_and_return_conditional_losses_36670040

inputs
inputs_1
identityW
addAddV2inputsinputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

E
)__inference_lambda_layer_call_fn_36670442

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lambda_layer_call_and_return_conditional_losses_366700552
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*½
serving_default©
O
feature_input>
serving_default_feature_input:0ÿÿÿÿÿÿÿÿÿR:
lambda0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ûî
þB
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
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
*u&call_and_return_all_conditional_losses
v_default_save_signature
w__call__"ý?
_tf_keras_networká?{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "feature_input"}, "name": "feature_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "value_conv", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value_conv", "inbound_nodes": [[["feature_input", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "value_reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}, "name": "value_reshape", "inbound_nodes": [[["value_conv", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["feature_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "value_dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value_dense", "inbound_nodes": [[["value_reshape", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}, "name": "reshape", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["value_dense", 0, 0, {}], ["reshape", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTmcAAAAAADijQKkAKQHaAXhyAQAA\nAHIBAAAA+kIvaG9tZS9zd2FuZy9wcm9qZWN0L1NtYXJ0V2Fsa2VyL1JMX1NpbXVsYXRpb24vY29k\nZS9hZ2VudF9tb2RlbHMucHnaCDxsYW1iZGE+RQAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "agent_models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["add", 0, 0, {}]]]}], "input_layers": [["feature_input", 0, 0]], "output_layers": [["lambda", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 8, 82]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "feature_input"}, "name": "feature_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "value_conv", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value_conv", "inbound_nodes": [[["feature_input", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "value_reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}, "name": "value_reshape", "inbound_nodes": [[["value_conv", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["feature_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "value_dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value_dense", "inbound_nodes": [[["value_reshape", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}, "name": "reshape", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["value_dense", 0, 0, {}], ["reshape", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTmcAAAAAADijQKkAKQHaAXhyAQAA\nAHIBAAAA+kIvaG9tZS9zd2FuZy9wcm9qZWN0L1NtYXJ0V2Fsa2VyL1JMX1NpbXVsYXRpb24vY29k\nZS9hZ2VudF9tb2RlbHMucHnaCDxsYW1iZGE+RQAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "agent_models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["add", 0, 0, {}]]]}], "input_layers": [["feature_input", 0, 0]], "output_layers": [["lambda", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
"
_tf_keras_input_layerâ{"class_name": "InputLayer", "name": "feature_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "feature_input"}}
ø	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*x&call_and_return_all_conditional_losses
y__call__"Ó
_tf_keras_layer¹{"class_name": "Conv2D", "name": "value_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value_conv", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 82}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 8, 82]}}
ú
	variables
regularization_losses
trainable_variables
	keras_api
*z&call_and_return_all_conditional_losses
{__call__"ë
_tf_keras_layerÑ{"class_name": "Reshape", "name": "value_reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value_reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}}
ð	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*|&call_and_return_all_conditional_losses
}__call__"Ë
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 82}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 8, 82]}}
÷

kernel
 bias
!	variables
"regularization_losses
#trainable_variables
$	keras_api
*~&call_and_return_all_conditional_losses
__call__"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "value_dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value_dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
ð
%	variables
&regularization_losses
'trainable_variables
(	keras_api
+&call_and_return_all_conditional_losses
__call__"ß
_tf_keras_layerÅ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}}
¥
)	variables
*regularization_losses
+trainable_variables
,	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerú{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 8]}]}
×
-	variables
.regularization_losses
/trainable_variables
0	keras_api
+&call_and_return_all_conditional_losses
__call__"Æ
_tf_keras_layer¬{"class_name": "Lambda", "name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTmcAAAAAADijQKkAKQHaAXhyAQAA\nAHIBAAAA+kIvaG9tZS9zd2FuZy9wcm9qZWN0L1NtYXJ0V2Fsa2VyL1JMX1NpbXVsYXRpb24vY29k\nZS9hZ2VudF9tb2RlbHMucHnaCDxsYW1iZGE+RQAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "agent_models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
¿
1iter

2beta_1

3beta_2
	4decay
5learning_ratemimjmkmlmm mnvovpvqvrvs vt"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
4
 5"
trackable_list_wrapper
J
0
1
2
3
4
 5"
trackable_list_wrapper
Ê

6layers
7layer_regularization_losses
8metrics
9non_trainable_variables

regularization_losses
:layer_metrics
trainable_variables
	variables
w__call__
v_default_save_signature
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
+:)R2value_conv/kernel
:2value_conv/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

;layers
<layer_regularization_losses
=metrics
>non_trainable_variables
	variables
regularization_losses
?layer_metrics
trainable_variables
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

@layers
Alayer_regularization_losses
Bmetrics
Cnon_trainable_variables
	variables
regularization_losses
Dlayer_metrics
trainable_variables
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
':%R2conv2d/kernel
:2conv2d/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

Elayers
Flayer_regularization_losses
Gmetrics
Hnon_trainable_variables
	variables
regularization_losses
Ilayer_metrics
trainable_variables
}__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
$:"2value_dense/kernel
:2value_dense/bias
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
­

Jlayers
Klayer_regularization_losses
Lmetrics
Mnon_trainable_variables
!	variables
"regularization_losses
Nlayer_metrics
#trainable_variables
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

Olayers
Player_regularization_losses
Qmetrics
Rnon_trainable_variables
%	variables
&regularization_losses
Slayer_metrics
'trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

Tlayers
Ulayer_regularization_losses
Vmetrics
Wnon_trainable_variables
)	variables
*regularization_losses
Xlayer_metrics
+trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

Ylayers
Zlayer_regularization_losses
[metrics
\non_trainable_variables
-	variables
.regularization_losses
]layer_metrics
/trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
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
.
^0
_1"
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
»
	`total
	acount
b	variables
c	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ÿ
	dtotal
	ecount
f
_fn_kwargs
g	variables
h	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
`0
a1"
trackable_list_wrapper
-
b	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
d0
e1"
trackable_list_wrapper
-
g	variables"
_generic_user_object
0:.R2Adam/value_conv/kernel/m
": 2Adam/value_conv/bias/m
,:*R2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
):'2Adam/value_dense/kernel/m
#:!2Adam/value_dense/bias/m
0:.R2Adam/value_conv/kernel/v
": 2Adam/value_conv/bias/v
,:*R2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
):'2Adam/value_dense/kernel/v
#:!2Adam/value_dense/bias/v
Ú2×
C__inference_model_layer_call_and_return_conditional_losses_36670098
C__inference_model_layer_call_and_return_conditional_losses_36670075
C__inference_model_layer_call_and_return_conditional_losses_36670288
C__inference_model_layer_call_and_return_conditional_losses_36670247À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
#__inference__wrapped_model_36669912Ä
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *4¢1
/,
feature_inputÿÿÿÿÿÿÿÿÿR
î2ë
(__inference_model_layer_call_fn_36670305
(__inference_model_layer_call_fn_36670322
(__inference_model_layer_call_fn_36670139
(__inference_model_layer_call_fn_36670179À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
H__inference_value_conv_layer_call_and_return_conditional_losses_36670332¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_value_conv_layer_call_fn_36670341¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_value_reshape_layer_call_and_return_conditional_losses_36670353¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_value_reshape_layer_call_fn_36670358¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_layer_call_and_return_conditional_losses_36670368¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_conv2d_layer_call_fn_36670377¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_value_dense_layer_call_and_return_conditional_losses_36670387¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_value_dense_layer_call_fn_36670396¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_reshape_layer_call_and_return_conditional_losses_36670408¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_reshape_layer_call_fn_36670413¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_add_layer_call_and_return_conditional_losses_36670419¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_add_layer_call_fn_36670425¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
D__inference_lambda_layer_call_and_return_conditional_losses_36670437
D__inference_lambda_layer_call_and_return_conditional_losses_36670431À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
)__inference_lambda_layer_call_fn_36670447
)__inference_lambda_layer_call_fn_36670442À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÓBÐ
&__inference_signature_wrapper_36670206feature_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
  
#__inference__wrapped_model_36669912y >¢;
4¢1
/,
feature_inputÿÿÿÿÿÿÿÿÿR
ª "/ª,
*
lambda 
lambdaÿÿÿÿÿÿÿÿÿÉ
A__inference_add_layer_call_and_return_conditional_losses_36670419Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
  
&__inference_add_layer_call_fn_36670425vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ´
D__inference_conv2d_layer_call_and_return_conditional_losses_36670368l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿR
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_layer_call_fn_36670377_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿR
ª " ÿÿÿÿÿÿÿÿÿ¨
D__inference_lambda_layer_call_and_return_conditional_losses_36670431`7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¨
D__inference_lambda_layer_call_and_return_conditional_losses_36670437`7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_lambda_layer_call_fn_36670442S7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

 
p
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_lambda_layer_call_fn_36670447S7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

 
p 
ª "ÿÿÿÿÿÿÿÿÿ¾
C__inference_model_layer_call_and_return_conditional_losses_36670075w F¢C
<¢9
/,
feature_inputÿÿÿÿÿÿÿÿÿR
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
C__inference_model_layer_call_and_return_conditional_losses_36670098w F¢C
<¢9
/,
feature_inputÿÿÿÿÿÿÿÿÿR
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
C__inference_model_layer_call_and_return_conditional_losses_36670247p ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿR
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
C__inference_model_layer_call_and_return_conditional_losses_36670288p ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿR
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_model_layer_call_fn_36670139j F¢C
<¢9
/,
feature_inputÿÿÿÿÿÿÿÿÿR
p

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_layer_call_fn_36670179j F¢C
<¢9
/,
feature_inputÿÿÿÿÿÿÿÿÿR
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_layer_call_fn_36670305c ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿR
p

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_layer_call_fn_36670322c ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿR
p 

 
ª "ÿÿÿÿÿÿÿÿÿ©
E__inference_reshape_layer_call_and_return_conditional_losses_36670408`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_reshape_layer_call_fn_36670413S7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿµ
&__inference_signature_wrapper_36670206 O¢L
¢ 
EªB
@
feature_input/,
feature_inputÿÿÿÿÿÿÿÿÿR"/ª,
*
lambda 
lambdaÿÿÿÿÿÿÿÿÿ¸
H__inference_value_conv_layer_call_and_return_conditional_losses_36670332l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿR
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_value_conv_layer_call_fn_36670341_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿR
ª " ÿÿÿÿÿÿÿÿÿ©
I__inference_value_dense_layer_call_and_return_conditional_losses_36670387\ /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_value_dense_layer_call_fn_36670396O /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¯
K__inference_value_reshape_layer_call_and_return_conditional_losses_36670353`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_value_reshape_layer_call_fn_36670358S7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ