??1
??
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
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??'

conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d/kernel
x
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*'
_output_shapes
:?*
dtype0
o
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:?*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@* 
shared_nameconv2d_1/kernel
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*'
_output_shapes
:?@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_3/gamma
?
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_3/beta
?
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_3/moving_mean
?
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_3/moving_variance
?
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma
?
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta
?
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean
?
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance
?
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma
?
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta
?
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean
?
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance
?
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_6/gamma
?
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_6/beta
?
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_6/moving_mean
?
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_6/moving_variance
?
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_7/gamma
?
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_7/beta
?
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_7/moving_mean
?
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_7/moving_variance
?
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:*
dtype0
?
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_8/gamma
?
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_8/beta
?
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:*
dtype0
?
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_8/moving_mean
?
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:*
dtype0
?
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_8/moving_variance
?
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ԟ
valueɟBş B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10

signatures
#_self_saveable_object_factories
	optimizer
loss
regularization_losses
trainable_variables
	variables
	keras_api
%
#_self_saveable_object_factories
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
?
 layer_with_weights-0
 layer-0
!layer_with_weights-1
!layer-1
"layer-2
#layer_with_weights-2
#layer-3
$layer_with_weights-3
$layer-4
%layer-5
&layer_with_weights-4
&layer-6
'layer_with_weights-5
'layer-7
#(_self_saveable_object_factories
)regularization_losses
*trainable_variables
+	variables
,	keras_api
4
#-_self_saveable_object_factories
.	keras_api
w
#/_self_saveable_object_factories
0regularization_losses
1trainable_variables
2	variables
3	keras_api
?
4layer_with_weights-0
4layer-0
5layer_with_weights-1
5layer-1
6layer-2
7layer_with_weights-2
7layer-3
8layer_with_weights-3
8layer-4
9layer-5
:layer_with_weights-4
:layer-6
;layer_with_weights-5
;layer-7
#<_self_saveable_object_factories
=regularization_losses
>trainable_variables
?	variables
@	keras_api
4
#A_self_saveable_object_factories
B	keras_api
w
#C_self_saveable_object_factories
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
?
Hlayer_with_weights-0
Hlayer-0
Ilayer_with_weights-1
Ilayer-1
Jlayer-2
#K_self_saveable_object_factories
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
4
#P_self_saveable_object_factories
Q	keras_api
w
#R_self_saveable_object_factories
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
 
 
 
 
 
 
?
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11
c12
d13
e14
f15
g16
h17
i18
j19
k20
l21
m22
n23
o24
p25
q26
r27
s28
t29
u30
v31
w32
x33
y34
z35
{36
|37
}38
~39
40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?
?layers
regularization_losses
 ?layer_regularization_losses
?layer_metrics
trainable_variables
?non_trainable_variables
	variables
?metrics
 
?

Wkernel
Xbias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis
	Ygamma
Zbeta
[moving_mean
\moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?

]kernel
^bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis
	_gamma
`beta
amoving_mean
bmoving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
 
 
V
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11
?
?layers
regularization_losses
 ?layer_regularization_losses
?layer_metrics
trainable_variables
?non_trainable_variables
	variables
?metrics
?

ckernel
dbias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis
	egamma
fbeta
gmoving_mean
hmoving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?

ikernel
jbias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis
	kgamma
lbeta
mmoving_mean
nmoving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?

okernel
pbias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis
	qgamma
rbeta
smoving_mean
tmoving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
 
 
?
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15
s16
t17
?
?layers
)regularization_losses
 ?layer_regularization_losses
?layer_metrics
*trainable_variables
?non_trainable_variables
+	variables
?metrics
 
 
 
 
 
 
?
?layers
0regularization_losses
 ?layer_regularization_losses
?layer_metrics
1trainable_variables
?non_trainable_variables
2	variables
?metrics
?

ukernel
vbias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis
	wgamma
xbeta
ymoving_mean
zmoving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?

{kernel
|bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis
	}gamma
~beta
moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
 
 
?
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
?11
?12
?13
?14
?15
?16
?17
?
?layers
=regularization_losses
 ?layer_regularization_losses
?layer_metrics
>trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 
 
 
?
?layers
Dregularization_losses
 ?layer_regularization_losses
?layer_metrics
Etrainable_variables
?non_trainable_variables
F	variables
?metrics
?
?kernel
	?bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
 
 
0
?0
?1
?2
?3
?4
?5
?
?layers
Lregularization_losses
 ?layer_regularization_losses
?layer_metrics
Mtrainable_variables
?non_trainable_variables
N	variables
?metrics
 
 
 
 
 
 
?
?layers
Sregularization_losses
 ?layer_regularization_losses
?layer_metrics
Ttrainable_variables
?non_trainable_variables
U	variables
?metrics
IG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbatch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEbatch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_2/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_2/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_2/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_2/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_2/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_2/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_3/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_3/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_3/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_3/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_4/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_4/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_4/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_4/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_4/moving_mean'variables/28/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_4/moving_variance'variables/29/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_5/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_5/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_5/gamma'variables/32/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_5/beta'variables/33/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_5/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_5/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_6/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_6/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_6/gamma'variables/38/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_6/beta'variables/39/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_6/moving_mean'variables/40/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_6/moving_variance'variables/41/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_7/kernel'variables/42/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_7/bias'variables/43/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_7/gamma'variables/44/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_7/beta'variables/45/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_7/moving_mean'variables/46/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_7/moving_variance'variables/47/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_8/kernel'variables/48/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_8/bias'variables/49/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_8/gamma'variables/50/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_8/beta'variables/51/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_8/moving_mean'variables/52/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_8/moving_variance'variables/53/.ATTRIBUTES/VARIABLE_VALUE
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
?
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11
c12
d13
e14
f15
g16
h17
i18
j19
k20
l21
m22
n23
o24
p25
q26
r27
s28
t29
u30
v31
w32
x33
y34
z35
{36
|37
}38
~39
40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
 
 
 
 

W0
X1
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 

Y0
Z1
[2
\3
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 

]0
^1
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 

_0
`1
a2
b3
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
*
0
1
2
3
4
5
 
 
V
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11
 
 
 
 

c0
d1
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 

e0
f1
g2
h3
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 

i0
j1
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 

k0
l1
m2
n3
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 

o0
p1
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 

q0
r1
s2
t3
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
8
 0
!1
"2
#3
$4
%5
&6
'7
 
 
?
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15
s16
t17
 
 
 
 
 
 
 
 
 

u0
v1
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 

w0
x1
y2
z3
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 

{0
|1
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 

}0
~1
2
?3
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 

?0
?1
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 
 
?0
?1
?2
?3
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
8
40
51
62
73
84
95
:6
;7
 
 
?
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
?11
?12
?13
?14
?15
?16
?17
 
 
 
 
 
 
 
 
 

?0
?1
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 
 
?0
?1
?2
?3
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
 
 
 
 
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics

H0
I1
J2
 
 
0
?0
?1
?2
?3
?4
?5
 
 
 
 
 
 
 
 
 

W0
X1
 
 
 
 

Y0
Z1
[2
\3
 
 
 
 
 
 
 
 
 

]0
^1
 
 
 
 

_0
`1
a2
b3
 
 
 
 
 
 
 
 
 

c0
d1
 
 
 
 

e0
f1
g2
h3
 
 
 
 
 
 
 
 
 

i0
j1
 
 
 
 

k0
l1
m2
n3
 
 
 
 
 
 
 
 
 

o0
p1
 
 
 
 

q0
r1
s2
t3
 
 
 
 

u0
v1
 
 
 
 

w0
x1
y2
z3
 
 
 
 
 
 
 
 
 

{0
|1
 
 
 
 

}0
~1
2
?3
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
?0
?1
?2
?3
 
 
 
 

?0
?1
 
 
 
 
 
?0
?1
?2
?3
 
 
 
 
 
 
?
serving_default_inputPlaceholder*0
_output_shapes
:??????????*
dtype0*%
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variance*B
Tin;
927*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_signature_wrapper_196694169
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOpConst*C
Tin<
:28*
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
"__inference__traced_save_196697022
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variance*B
Tin;
927*
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
%__inference__traced_restore_196697194??$
?
?
9__inference_batch_normalization_4_layer_call_fn_196696160

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1966912962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196695903

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196695994

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
e
I__inference_activation_layer_call_and_return_conditional_losses_196695628

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_3_layer_call_and_return_conditional_losses_196691462

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
g
K__inference_activation_3_layer_call_and_return_conditional_losses_196696087

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196691605

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
E__inference_conv2d_layer_call_and_return_conditional_losses_196695490

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_activation_1_layer_call_and_return_conditional_losses_196695781

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196696443

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?"
?
I__inference_input_conv_layer_call_and_return_conditional_losses_196690911
conv2d_input
conv2d_196690880
conv2d_196690882!
batch_normalization_196690885!
batch_normalization_196690887!
batch_normalization_196690889!
batch_normalization_196690891
conv2d_1_196690895
conv2d_1_196690897#
batch_normalization_1_196690900#
batch_normalization_1_196690902#
batch_normalization_1_196690904#
batch_normalization_1_196690906
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_196690880conv2d_196690882*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_1966906662 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_196690885batch_normalization_196690887batch_normalization_196690889batch_normalization_196690891*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1966907172-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_layer_call_and_return_conditional_losses_1966907582
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_196690895conv2d_1_196690897*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_1966907762"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_196690900batch_normalization_1_196690902batch_normalization_1_196690904batch_normalization_1_196690906*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1966908272/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_1_layer_call_and_return_conditional_losses_1966908682
activation_1/PartitionedCall?
IdentityIdentity%activation_1/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:^ Z
0
_output_shapes
:??????????
&
_user_specified_nameconv2d_input
?
?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196696352

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_7_layer_call_fn_196696622

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1966921762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
F
*__inference_relu_0_layer_call_fn_196695163

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_relu_0_layer_call_and_return_conditional_losses_1966932552
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?/
?
J__inference_res_block_1_layer_call_and_return_conditional_losses_196692509
conv2d_5_input
conv2d_5_196692212
conv2d_5_196692214#
batch_normalization_5_196692279#
batch_normalization_5_196692281#
batch_normalization_5_196692283#
batch_normalization_5_196692285
conv2d_6_196692322
conv2d_6_196692324#
batch_normalization_6_196692389#
batch_normalization_6_196692391#
batch_normalization_6_196692393#
batch_normalization_6_196692395
conv2d_7_196692432
conv2d_7_196692434#
batch_normalization_7_196692499#
batch_normalization_7_196692501#
batch_normalization_7_196692503#
batch_normalization_7_196692505
identity??-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_196692212conv2d_5_196692214*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_1966922012"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_196692279batch_normalization_5_196692281batch_normalization_5_196692283batch_normalization_5_196692285*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1966922342/
-batch_normalization_5/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_4_layer_call_and_return_conditional_losses_1966922932
activation_4/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_196692322conv2d_6_196692324*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_1966923112"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_196692389batch_normalization_6_196692391batch_normalization_6_196692393batch_normalization_6_196692395*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1966923442/
-batch_normalization_6/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_5_layer_call_and_return_conditional_losses_1966924032
activation_5/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_196692432conv2d_7_196692434*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_7_layer_call_and_return_conditional_losses_1966924212"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_196692499batch_normalization_7_196692501batch_normalization_7_196692503batch_normalization_7_196692505*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1966924542/
-batch_normalization_7/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_7/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_5_input
?
?
F__inference_feature_layer_call_and_return_conditional_losses_196695437

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R*
paddingVALID*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R2
conv2d_8/BiasAdd?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3?
activation_6/ReluRelu*batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R2
activation_6/Relu?
IdentityIdentityactivation_6/Relu:activations:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1 ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
+__inference_feature_layer_call_fn_196695454

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
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_feature_layer_call_and_return_conditional_losses_1966929922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_7_layer_call_and_return_conditional_losses_196696551

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_1_layer_call_and_return_conditional_losses_196695643

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_3_layer_call_fn_196695958

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
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_1966914622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
.__inference_input_conv_layer_call_fn_196690975
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_input_conv_layer_call_and_return_conditional_losses_1966909482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:??????????
&
_user_specified_nameconv2d_input
?
L
0__inference_activation_1_layer_call_fn_196695786

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_1_layer_call_and_return_conditional_losses_1966908682
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
,__inference_conv2d_2_layer_call_fn_196695805

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
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_1966913522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196695823

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196695597

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
g
K__inference_activation_2_layer_call_and_return_conditional_losses_196691444

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_8_layer_call_fn_196696827

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1966928252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?;
?
D__inference_model_layer_call_and_return_conditional_losses_196693943

inputs
input_conv_196693823
input_conv_196693825
input_conv_196693827
input_conv_196693829
input_conv_196693831
input_conv_196693833
input_conv_196693835
input_conv_196693837
input_conv_196693839
input_conv_196693841
input_conv_196693843
input_conv_196693845
res_block_0_196693848
res_block_0_196693850
res_block_0_196693852
res_block_0_196693854
res_block_0_196693856
res_block_0_196693858
res_block_0_196693860
res_block_0_196693862
res_block_0_196693864
res_block_0_196693866
res_block_0_196693868
res_block_0_196693870
res_block_0_196693872
res_block_0_196693874
res_block_0_196693876
res_block_0_196693878
res_block_0_196693880
res_block_0_196693882
res_block_1_196693887
res_block_1_196693889
res_block_1_196693891
res_block_1_196693893
res_block_1_196693895
res_block_1_196693897
res_block_1_196693899
res_block_1_196693901
res_block_1_196693903
res_block_1_196693905
res_block_1_196693907
res_block_1_196693909
res_block_1_196693911
res_block_1_196693913
res_block_1_196693915
res_block_1_196693917
res_block_1_196693919
res_block_1_196693921
feature_196693926
feature_196693928
feature_196693930
feature_196693932
feature_196693934
feature_196693936
identity??feature/StatefulPartitionedCall?"input_conv/StatefulPartitionedCall?#res_block_0/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_196693823input_conv_196693825input_conv_196693827input_conv_196693829input_conv_196693831input_conv_196693833input_conv_196693835input_conv_196693837input_conv_196693839input_conv_196693841input_conv_196693843input_conv_196693845*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_input_conv_layer_call_and_return_conditional_losses_1966910112$
"input_conv/StatefulPartitionedCall?
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_196693848res_block_0_196693850res_block_0_196693852res_block_0_196693854res_block_0_196693856res_block_0_196693858res_block_0_196693860res_block_0_196693862res_block_0_196693864res_block_0_196693866res_block_0_196693868res_block_0_196693870res_block_0_196693872res_block_0_196693874res_block_0_196693876res_block_0_196693878res_block_0_196693880res_block_0_196693882*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_0_layer_call_and_return_conditional_losses_1966918482%
#res_block_0/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV2+input_conv/StatefulPartitionedCall:output:0,res_block_0/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add/AddV2?
relu_0/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_relu_0_layer_call_and_return_conditional_losses_1966932552
relu_0/PartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_196693887res_block_1_196693889res_block_1_196693891res_block_1_196693893res_block_1_196693895res_block_1_196693897res_block_1_196693899res_block_1_196693901res_block_1_196693903res_block_1_196693905res_block_1_196693907res_block_1_196693909res_block_1_196693911res_block_1_196693913res_block_1_196693915res_block_1_196693917res_block_1_196693919res_block_1_196693921*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_1_layer_call_and_return_conditional_losses_1966926972%
#res_block_1/StatefulPartitionedCall?
tf.__operators__.add_1/AddV2AddV2relu_0/PartitionedCall:output:0,res_block_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add_1/AddV2?
relu_1/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_relu_1_layer_call_and_return_conditional_losses_1966933882
relu_1/PartitionedCall?
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_196693926feature_196693928feature_196693930feature_196693932feature_196693934feature_196693936*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_feature_layer_call_and_return_conditional_losses_1966930282!
feature/StatefulPartitionedCall?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose(feature/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????R2"
 tf.compat.v1.transpose/transpose?
feature_linear/PartitionedCallPartitionedCall$tf.compat.v1.transpose/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_feature_linear_layer_call_and_return_conditional_losses_1966934492 
feature_linear/PartitionedCall?
IdentityIdentity'feature_linear/PartitionedCall:output:0 ^feature/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^res_block_0/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
feature/StatefulPartitionedCallfeature/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#res_block_0/StatefulPartitionedCall#res_block_0/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196695841

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
e
I__inference_activation_layer_call_and_return_conditional_losses_196690758

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
L
0__inference_activation_6_layer_call_fn_196696837

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_6_layer_call_and_return_conditional_losses_1966929422
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_1_layer_call_fn_196695701

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1966906102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196691976

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196692901

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196696191

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?V
?
J__inference_res_block_0_layer_call_and_return_conditional_losses_196695005

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource
identity??5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_2/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3?
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_2/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dactivation_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_3/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3?
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_3/Relu?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_4/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3?
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
??
?3
D__inference_model_layer_call_and_return_conditional_losses_196694563

inputs4
0input_conv_conv2d_conv2d_readvariableop_resource5
1input_conv_conv2d_biasadd_readvariableop_resource:
6input_conv_batch_normalization_readvariableop_resource<
8input_conv_batch_normalization_readvariableop_1_resourceK
Ginput_conv_batch_normalization_fusedbatchnormv3_readvariableop_resourceM
Iinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource6
2input_conv_conv2d_1_conv2d_readvariableop_resource7
3input_conv_conv2d_1_biasadd_readvariableop_resource<
8input_conv_batch_normalization_1_readvariableop_resource>
:input_conv_batch_normalization_1_readvariableop_1_resourceM
Iinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceO
Kinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7
3res_block_0_conv2d_2_conv2d_readvariableop_resource8
4res_block_0_conv2d_2_biasadd_readvariableop_resource=
9res_block_0_batch_normalization_2_readvariableop_resource?
;res_block_0_batch_normalization_2_readvariableop_1_resourceN
Jres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceP
Lres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7
3res_block_0_conv2d_3_conv2d_readvariableop_resource8
4res_block_0_conv2d_3_biasadd_readvariableop_resource=
9res_block_0_batch_normalization_3_readvariableop_resource?
;res_block_0_batch_normalization_3_readvariableop_1_resourceN
Jres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3res_block_0_conv2d_4_conv2d_readvariableop_resource8
4res_block_0_conv2d_4_biasadd_readvariableop_resource=
9res_block_0_batch_normalization_4_readvariableop_resource?
;res_block_0_batch_normalization_4_readvariableop_1_resourceN
Jres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceP
Lres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3res_block_1_conv2d_5_conv2d_readvariableop_resource8
4res_block_1_conv2d_5_biasadd_readvariableop_resource=
9res_block_1_batch_normalization_5_readvariableop_resource?
;res_block_1_batch_normalization_5_readvariableop_1_resourceN
Jres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7
3res_block_1_conv2d_6_conv2d_readvariableop_resource8
4res_block_1_conv2d_6_biasadd_readvariableop_resource=
9res_block_1_batch_normalization_6_readvariableop_resource?
;res_block_1_batch_normalization_6_readvariableop_1_resourceN
Jres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceP
Lres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7
3res_block_1_conv2d_7_conv2d_readvariableop_resource8
4res_block_1_conv2d_7_biasadd_readvariableop_resource=
9res_block_1_batch_normalization_7_readvariableop_resource?
;res_block_1_batch_normalization_7_readvariableop_1_resourceN
Jres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceP
Lres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource3
/feature_conv2d_8_conv2d_readvariableop_resource4
0feature_conv2d_8_biasadd_readvariableop_resource9
5feature_batch_normalization_8_readvariableop_resource;
7feature_batch_normalization_8_readvariableop_1_resourceJ
Ffeature_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceL
Hfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp??feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?,feature/batch_normalization_8/ReadVariableOp?.feature/batch_normalization_8/ReadVariableOp_1?'feature/conv2d_8/BiasAdd/ReadVariableOp?&feature/conv2d_8/Conv2D/ReadVariableOp?>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp?@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-input_conv/batch_normalization/ReadVariableOp?/input_conv/batch_normalization/ReadVariableOp_1?@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?/input_conv/batch_normalization_1/ReadVariableOp?1input_conv/batch_normalization_1/ReadVariableOp_1?(input_conv/conv2d/BiasAdd/ReadVariableOp?'input_conv/conv2d/Conv2D/ReadVariableOp?*input_conv/conv2d_1/BiasAdd/ReadVariableOp?)input_conv/conv2d_1/Conv2D/ReadVariableOp?Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?0res_block_0/batch_normalization_2/ReadVariableOp?2res_block_0/batch_normalization_2/ReadVariableOp_1?Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?0res_block_0/batch_normalization_3/ReadVariableOp?2res_block_0/batch_normalization_3/ReadVariableOp_1?Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?0res_block_0/batch_normalization_4/ReadVariableOp?2res_block_0/batch_normalization_4/ReadVariableOp_1?+res_block_0/conv2d_2/BiasAdd/ReadVariableOp?*res_block_0/conv2d_2/Conv2D/ReadVariableOp?+res_block_0/conv2d_3/BiasAdd/ReadVariableOp?*res_block_0/conv2d_3/Conv2D/ReadVariableOp?+res_block_0/conv2d_4/BiasAdd/ReadVariableOp?*res_block_0/conv2d_4/Conv2D/ReadVariableOp?Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?0res_block_1/batch_normalization_5/ReadVariableOp?2res_block_1/batch_normalization_5/ReadVariableOp_1?Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?0res_block_1/batch_normalization_6/ReadVariableOp?2res_block_1/batch_normalization_6/ReadVariableOp_1?Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?0res_block_1/batch_normalization_7/ReadVariableOp?2res_block_1/batch_normalization_7/ReadVariableOp_1?+res_block_1/conv2d_5/BiasAdd/ReadVariableOp?*res_block_1/conv2d_5/Conv2D/ReadVariableOp?+res_block_1/conv2d_6/BiasAdd/ReadVariableOp?*res_block_1/conv2d_6/Conv2D/ReadVariableOp?+res_block_1/conv2d_7/BiasAdd/ReadVariableOp?*res_block_1/conv2d_7/Conv2D/ReadVariableOp?
'input_conv/conv2d/Conv2D/ReadVariableOpReadVariableOp0input_conv_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02)
'input_conv/conv2d/Conv2D/ReadVariableOp?
input_conv/conv2d/Conv2DConv2Dinputs/input_conv/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
input_conv/conv2d/Conv2D?
(input_conv/conv2d/BiasAdd/ReadVariableOpReadVariableOp1input_conv_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(input_conv/conv2d/BiasAdd/ReadVariableOp?
input_conv/conv2d/BiasAddBiasAdd!input_conv/conv2d/Conv2D:output:00input_conv/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
input_conv/conv2d/BiasAdd?
-input_conv/batch_normalization/ReadVariableOpReadVariableOp6input_conv_batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-input_conv/batch_normalization/ReadVariableOp?
/input_conv/batch_normalization/ReadVariableOp_1ReadVariableOp8input_conv_batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/input_conv/batch_normalization/ReadVariableOp_1?
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/input_conv/batch_normalization/FusedBatchNormV3FusedBatchNormV3"input_conv/conv2d/BiasAdd:output:05input_conv/batch_normalization/ReadVariableOp:value:07input_conv/batch_normalization/ReadVariableOp_1:value:0Finput_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hinput_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/input_conv/batch_normalization/FusedBatchNormV3?
input_conv/activation/ReluRelu3input_conv/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
input_conv/activation/Relu?
)input_conv/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2input_conv_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02+
)input_conv/conv2d_1/Conv2D/ReadVariableOp?
input_conv/conv2d_1/Conv2DConv2D(input_conv/activation/Relu:activations:01input_conv/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingVALID*
strides
2
input_conv/conv2d_1/Conv2D?
*input_conv/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3input_conv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*input_conv/conv2d_1/BiasAdd/ReadVariableOp?
input_conv/conv2d_1/BiasAddBiasAdd#input_conv/conv2d_1/Conv2D:output:02input_conv/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
input_conv/conv2d_1/BiasAdd?
/input_conv/batch_normalization_1/ReadVariableOpReadVariableOp8input_conv_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype021
/input_conv/batch_normalization_1/ReadVariableOp?
1input_conv/batch_normalization_1/ReadVariableOp_1ReadVariableOp:input_conv_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1input_conv/batch_normalization_1/ReadVariableOp_1?
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02B
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
1input_conv/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$input_conv/conv2d_1/BiasAdd:output:07input_conv/batch_normalization_1/ReadVariableOp:value:09input_conv/batch_normalization_1/ReadVariableOp_1:value:0Hinput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jinput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 23
1input_conv/batch_normalization_1/FusedBatchNormV3?
input_conv/activation_1/ReluRelu5input_conv/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
input_conv/activation_1/Relu?
*res_block_0/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_2/Conv2D/ReadVariableOp?
res_block_0/conv2d_2/Conv2DConv2D*input_conv/activation_1/Relu:activations:02res_block_0/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_0/conv2d_2/Conv2D?
+res_block_0/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_2/BiasAdd/ReadVariableOp?
res_block_0/conv2d_2/BiasAddBiasAdd$res_block_0/conv2d_2/Conv2D:output:03res_block_0/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/conv2d_2/BiasAdd?
0res_block_0/batch_normalization_2/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_2/ReadVariableOp?
2res_block_0/batch_normalization_2/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_2/ReadVariableOp_1?
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
2res_block_0/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_2/BiasAdd:output:08res_block_0/batch_normalization_2/ReadVariableOp:value:0:res_block_0/batch_normalization_2/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_0/batch_normalization_2/FusedBatchNormV3?
res_block_0/activation_2/ReluRelu6res_block_0/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/activation_2/Relu?
*res_block_0/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_3/Conv2D/ReadVariableOp?
res_block_0/conv2d_3/Conv2DConv2D+res_block_0/activation_2/Relu:activations:02res_block_0/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_0/conv2d_3/Conv2D?
+res_block_0/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_3/BiasAdd/ReadVariableOp?
res_block_0/conv2d_3/BiasAddBiasAdd$res_block_0/conv2d_3/Conv2D:output:03res_block_0/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/conv2d_3/BiasAdd?
0res_block_0/batch_normalization_3/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_3/ReadVariableOp?
2res_block_0/batch_normalization_3/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_3/ReadVariableOp_1?
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
2res_block_0/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_3/BiasAdd:output:08res_block_0/batch_normalization_3/ReadVariableOp:value:0:res_block_0/batch_normalization_3/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_0/batch_normalization_3/FusedBatchNormV3?
res_block_0/activation_3/ReluRelu6res_block_0/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/activation_3/Relu?
*res_block_0/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_4/Conv2D/ReadVariableOp?
res_block_0/conv2d_4/Conv2DConv2D+res_block_0/activation_3/Relu:activations:02res_block_0/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_0/conv2d_4/Conv2D?
+res_block_0/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_4/BiasAdd/ReadVariableOp?
res_block_0/conv2d_4/BiasAddBiasAdd$res_block_0/conv2d_4/Conv2D:output:03res_block_0/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/conv2d_4/BiasAdd?
0res_block_0/batch_normalization_4/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_4/ReadVariableOp?
2res_block_0/batch_normalization_4/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_4/ReadVariableOp_1?
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
2res_block_0/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_4/BiasAdd:output:08res_block_0/batch_normalization_4/ReadVariableOp:value:0:res_block_0/batch_normalization_4/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_0/batch_normalization_4/FusedBatchNormV3?
tf.__operators__.add/AddV2AddV2*input_conv/activation_1/Relu:activations:06res_block_0/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add/AddV2|
relu_0/ReluRelutf.__operators__.add/AddV2:z:0*
T0*/
_output_shapes
:?????????R@2
relu_0/Relu?
*res_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_5/Conv2D/ReadVariableOp?
res_block_1/conv2d_5/Conv2DConv2Drelu_0/Relu:activations:02res_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_1/conv2d_5/Conv2D?
+res_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_5/BiasAdd/ReadVariableOp?
res_block_1/conv2d_5/BiasAddBiasAdd$res_block_1/conv2d_5/Conv2D:output:03res_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/conv2d_5/BiasAdd?
0res_block_1/batch_normalization_5/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_5/ReadVariableOp?
2res_block_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_5/ReadVariableOp_1?
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
2res_block_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_5/BiasAdd:output:08res_block_1/batch_normalization_5/ReadVariableOp:value:0:res_block_1/batch_normalization_5/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_1/batch_normalization_5/FusedBatchNormV3?
res_block_1/activation_4/ReluRelu6res_block_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/activation_4/Relu?
*res_block_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_6/Conv2D/ReadVariableOp?
res_block_1/conv2d_6/Conv2DConv2D+res_block_1/activation_4/Relu:activations:02res_block_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_1/conv2d_6/Conv2D?
+res_block_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_6/BiasAdd/ReadVariableOp?
res_block_1/conv2d_6/BiasAddBiasAdd$res_block_1/conv2d_6/Conv2D:output:03res_block_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/conv2d_6/BiasAdd?
0res_block_1/batch_normalization_6/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_6/ReadVariableOp?
2res_block_1/batch_normalization_6/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_6/ReadVariableOp_1?
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
2res_block_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_6/BiasAdd:output:08res_block_1/batch_normalization_6/ReadVariableOp:value:0:res_block_1/batch_normalization_6/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_1/batch_normalization_6/FusedBatchNormV3?
res_block_1/activation_5/ReluRelu6res_block_1/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/activation_5/Relu?
*res_block_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_7/Conv2D/ReadVariableOp?
res_block_1/conv2d_7/Conv2DConv2D+res_block_1/activation_5/Relu:activations:02res_block_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_1/conv2d_7/Conv2D?
+res_block_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_7/BiasAdd/ReadVariableOp?
res_block_1/conv2d_7/BiasAddBiasAdd$res_block_1/conv2d_7/Conv2D:output:03res_block_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/conv2d_7/BiasAdd?
0res_block_1/batch_normalization_7/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_7/ReadVariableOp?
2res_block_1/batch_normalization_7/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_7/ReadVariableOp_1?
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
2res_block_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_7/BiasAdd:output:08res_block_1/batch_normalization_7/ReadVariableOp:value:0:res_block_1/batch_normalization_7/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_1/batch_normalization_7/FusedBatchNormV3?
tf.__operators__.add_1/AddV2AddV2relu_0/Relu:activations:06res_block_1/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add_1/AddV2~
relu_1/ReluRelu tf.__operators__.add_1/AddV2:z:0*
T0*/
_output_shapes
:?????????R@2
relu_1/Relu?
&feature/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/feature_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&feature/conv2d_8/Conv2D/ReadVariableOp?
feature/conv2d_8/Conv2DConv2Drelu_1/Relu:activations:0.feature/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R*
paddingVALID*
strides
2
feature/conv2d_8/Conv2D?
'feature/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0feature_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'feature/conv2d_8/BiasAdd/ReadVariableOp?
feature/conv2d_8/BiasAddBiasAdd feature/conv2d_8/Conv2D:output:0/feature/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R2
feature/conv2d_8/BiasAdd?
,feature/batch_normalization_8/ReadVariableOpReadVariableOp5feature_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02.
,feature/batch_normalization_8/ReadVariableOp?
.feature/batch_normalization_8/ReadVariableOp_1ReadVariableOp7feature_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype020
.feature/batch_normalization_8/ReadVariableOp_1?
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
.feature/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3!feature/conv2d_8/BiasAdd:output:04feature/batch_normalization_8/ReadVariableOp:value:06feature/batch_normalization_8/ReadVariableOp_1:value:0Efeature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gfeature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 20
.feature/batch_normalization_8/FusedBatchNormV3?
feature/activation_6/ReluRelu2feature/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R2
feature/activation_6/Relu?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose'feature/activation_6/Relu:activations:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????R2"
 tf.compat.v1.transpose/transpose?
IdentityIdentity$tf.compat.v1.transpose/transpose:y:0>^feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp@^feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1-^feature/batch_normalization_8/ReadVariableOp/^feature/batch_normalization_8/ReadVariableOp_1(^feature/conv2d_8/BiasAdd/ReadVariableOp'^feature/conv2d_8/Conv2D/ReadVariableOp?^input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpA^input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^input_conv/batch_normalization/ReadVariableOp0^input_conv/batch_normalization/ReadVariableOp_1A^input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^input_conv/batch_normalization_1/ReadVariableOp2^input_conv/batch_normalization_1/ReadVariableOp_1)^input_conv/conv2d/BiasAdd/ReadVariableOp(^input_conv/conv2d/Conv2D/ReadVariableOp+^input_conv/conv2d_1/BiasAdd/ReadVariableOp*^input_conv/conv2d_1/Conv2D/ReadVariableOpB^res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_2/ReadVariableOp3^res_block_0/batch_normalization_2/ReadVariableOp_1B^res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_3/ReadVariableOp3^res_block_0/batch_normalization_3/ReadVariableOp_1B^res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_4/ReadVariableOp3^res_block_0/batch_normalization_4/ReadVariableOp_1,^res_block_0/conv2d_2/BiasAdd/ReadVariableOp+^res_block_0/conv2d_2/Conv2D/ReadVariableOp,^res_block_0/conv2d_3/BiasAdd/ReadVariableOp+^res_block_0/conv2d_3/Conv2D/ReadVariableOp,^res_block_0/conv2d_4/BiasAdd/ReadVariableOp+^res_block_0/conv2d_4/Conv2D/ReadVariableOpB^res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_5/ReadVariableOp3^res_block_1/batch_normalization_5/ReadVariableOp_1B^res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_6/ReadVariableOp3^res_block_1/batch_normalization_6/ReadVariableOp_1B^res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_7/ReadVariableOp3^res_block_1/batch_normalization_7/ReadVariableOp_1,^res_block_1/conv2d_5/BiasAdd/ReadVariableOp+^res_block_1/conv2d_5/Conv2D/ReadVariableOp,^res_block_1/conv2d_6/BiasAdd/ReadVariableOp+^res_block_1/conv2d_6/Conv2D/ReadVariableOp,^res_block_1/conv2d_7/BiasAdd/ReadVariableOp+^res_block_1/conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::2~
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12\
,feature/batch_normalization_8/ReadVariableOp,feature/batch_normalization_8/ReadVariableOp2`
.feature/batch_normalization_8/ReadVariableOp_1.feature/batch_normalization_8/ReadVariableOp_12R
'feature/conv2d_8/BiasAdd/ReadVariableOp'feature/conv2d_8/BiasAdd/ReadVariableOp2P
&feature/conv2d_8/Conv2D/ReadVariableOp&feature/conv2d_8/Conv2D/ReadVariableOp2?
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-input_conv/batch_normalization/ReadVariableOp-input_conv/batch_normalization/ReadVariableOp2b
/input_conv/batch_normalization/ReadVariableOp_1/input_conv/batch_normalization/ReadVariableOp_12?
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12b
/input_conv/batch_normalization_1/ReadVariableOp/input_conv/batch_normalization_1/ReadVariableOp2f
1input_conv/batch_normalization_1/ReadVariableOp_11input_conv/batch_normalization_1/ReadVariableOp_12T
(input_conv/conv2d/BiasAdd/ReadVariableOp(input_conv/conv2d/BiasAdd/ReadVariableOp2R
'input_conv/conv2d/Conv2D/ReadVariableOp'input_conv/conv2d/Conv2D/ReadVariableOp2X
*input_conv/conv2d_1/BiasAdd/ReadVariableOp*input_conv/conv2d_1/BiasAdd/ReadVariableOp2V
)input_conv/conv2d_1/Conv2D/ReadVariableOp)input_conv/conv2d_1/Conv2D/ReadVariableOp2?
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2?
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_2/ReadVariableOp0res_block_0/batch_normalization_2/ReadVariableOp2h
2res_block_0/batch_normalization_2/ReadVariableOp_12res_block_0/batch_normalization_2/ReadVariableOp_12?
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_3/ReadVariableOp0res_block_0/batch_normalization_3/ReadVariableOp2h
2res_block_0/batch_normalization_3/ReadVariableOp_12res_block_0/batch_normalization_3/ReadVariableOp_12?
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_4/ReadVariableOp0res_block_0/batch_normalization_4/ReadVariableOp2h
2res_block_0/batch_normalization_4/ReadVariableOp_12res_block_0/batch_normalization_4/ReadVariableOp_12Z
+res_block_0/conv2d_2/BiasAdd/ReadVariableOp+res_block_0/conv2d_2/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_2/Conv2D/ReadVariableOp*res_block_0/conv2d_2/Conv2D/ReadVariableOp2Z
+res_block_0/conv2d_3/BiasAdd/ReadVariableOp+res_block_0/conv2d_3/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_3/Conv2D/ReadVariableOp*res_block_0/conv2d_3/Conv2D/ReadVariableOp2Z
+res_block_0/conv2d_4/BiasAdd/ReadVariableOp+res_block_0/conv2d_4/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_4/Conv2D/ReadVariableOp*res_block_0/conv2d_4/Conv2D/ReadVariableOp2?
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_5/ReadVariableOp0res_block_1/batch_normalization_5/ReadVariableOp2h
2res_block_1/batch_normalization_5/ReadVariableOp_12res_block_1/batch_normalization_5/ReadVariableOp_12?
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_6/ReadVariableOp0res_block_1/batch_normalization_6/ReadVariableOp2h
2res_block_1/batch_normalization_6/ReadVariableOp_12res_block_1/batch_normalization_6/ReadVariableOp_12?
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_7/ReadVariableOp0res_block_1/batch_normalization_7/ReadVariableOp2h
2res_block_1/batch_normalization_7/ReadVariableOp_12res_block_1/batch_normalization_7/ReadVariableOp_12Z
+res_block_1/conv2d_5/BiasAdd/ReadVariableOp+res_block_1/conv2d_5/BiasAdd/ReadVariableOp2X
*res_block_1/conv2d_5/Conv2D/ReadVariableOp*res_block_1/conv2d_5/Conv2D/ReadVariableOp2Z
+res_block_1/conv2d_6/BiasAdd/ReadVariableOp+res_block_1/conv2d_6/BiasAdd/ReadVariableOp2X
*res_block_1/conv2d_6/Conv2D/ReadVariableOp*res_block_1/conv2d_6/Conv2D/ReadVariableOp2Z
+res_block_1/conv2d_7/BiasAdd/ReadVariableOp+res_block_1/conv2d_7/BiasAdd/ReadVariableOp2X
*res_block_1/conv2d_7/Conv2D/ReadVariableOp*res_block_1/conv2d_7/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
I__inference_input_conv_layer_call_and_return_conditional_losses_196690877
conv2d_input
conv2d_196690677
conv2d_196690679!
batch_normalization_196690744!
batch_normalization_196690746!
batch_normalization_196690748!
batch_normalization_196690750
conv2d_1_196690787
conv2d_1_196690789#
batch_normalization_1_196690854#
batch_normalization_1_196690856#
batch_normalization_1_196690858#
batch_normalization_1_196690860
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_196690677conv2d_196690679*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_1966906662 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_196690744batch_normalization_196690746batch_normalization_196690748batch_normalization_196690750*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1966906992-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_layer_call_and_return_conditional_losses_1966907582
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_196690787conv2d_1_196690789*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_1966907762"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_196690854batch_normalization_1_196690856batch_normalization_1_196690858batch_normalization_1_196690860*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1966908092/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_1_layer_call_and_return_conditional_losses_1966908682
activation_1/PartitionedCall?
IdentityIdentity%activation_1/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:^ Z
0
_output_shapes
:??????????
&
_user_specified_nameconv2d_input
?
?
/__inference_res_block_1_layer_call_fn_196692736
conv2d_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_1_layer_call_and_return_conditional_losses_1966926972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_5_input
?
?
9__inference_batch_normalization_8_layer_call_fn_196696752

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1966928832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?!
?
I__inference_input_conv_layer_call_and_return_conditional_losses_196691011

inputs
conv2d_196690980
conv2d_196690982!
batch_normalization_196690985!
batch_normalization_196690987!
batch_normalization_196690989!
batch_normalization_196690991
conv2d_1_196690995
conv2d_1_196690997#
batch_normalization_1_196691000#
batch_normalization_1_196691002#
batch_normalization_1_196691004#
batch_normalization_1_196691006
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_196690980conv2d_196690982*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_1966906662 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_196690985batch_normalization_196690987batch_normalization_196690989batch_normalization_196690991*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1966907172-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_layer_call_and_return_conditional_losses_1966907582
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_196690995conv2d_1_196690997*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_1966907762"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_196691000batch_normalization_1_196691002batch_normalization_1_196691004batch_normalization_1_196691006*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1966908272/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_1_layer_call_and_return_conditional_losses_1966908682
activation_1/PartitionedCall?
IdentityIdentity%activation_1/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?;
?
D__inference_model_layer_call_and_return_conditional_losses_196693707

inputs
input_conv_196693587
input_conv_196693589
input_conv_196693591
input_conv_196693593
input_conv_196693595
input_conv_196693597
input_conv_196693599
input_conv_196693601
input_conv_196693603
input_conv_196693605
input_conv_196693607
input_conv_196693609
res_block_0_196693612
res_block_0_196693614
res_block_0_196693616
res_block_0_196693618
res_block_0_196693620
res_block_0_196693622
res_block_0_196693624
res_block_0_196693626
res_block_0_196693628
res_block_0_196693630
res_block_0_196693632
res_block_0_196693634
res_block_0_196693636
res_block_0_196693638
res_block_0_196693640
res_block_0_196693642
res_block_0_196693644
res_block_0_196693646
res_block_1_196693651
res_block_1_196693653
res_block_1_196693655
res_block_1_196693657
res_block_1_196693659
res_block_1_196693661
res_block_1_196693663
res_block_1_196693665
res_block_1_196693667
res_block_1_196693669
res_block_1_196693671
res_block_1_196693673
res_block_1_196693675
res_block_1_196693677
res_block_1_196693679
res_block_1_196693681
res_block_1_196693683
res_block_1_196693685
feature_196693690
feature_196693692
feature_196693694
feature_196693696
feature_196693698
feature_196693700
identity??feature/StatefulPartitionedCall?"input_conv/StatefulPartitionedCall?#res_block_0/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_196693587input_conv_196693589input_conv_196693591input_conv_196693593input_conv_196693595input_conv_196693597input_conv_196693599input_conv_196693601input_conv_196693603input_conv_196693605input_conv_196693607input_conv_196693609*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_input_conv_layer_call_and_return_conditional_losses_1966909482$
"input_conv/StatefulPartitionedCall?
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_196693612res_block_0_196693614res_block_0_196693616res_block_0_196693618res_block_0_196693620res_block_0_196693622res_block_0_196693624res_block_0_196693626res_block_0_196693628res_block_0_196693630res_block_0_196693632res_block_0_196693634res_block_0_196693636res_block_0_196693638res_block_0_196693640res_block_0_196693642res_block_0_196693644res_block_0_196693646*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_0_layer_call_and_return_conditional_losses_1966917592%
#res_block_0/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV2+input_conv/StatefulPartitionedCall:output:0,res_block_0/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add/AddV2?
relu_0/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_relu_0_layer_call_and_return_conditional_losses_1966932552
relu_0/PartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_196693651res_block_1_196693653res_block_1_196693655res_block_1_196693657res_block_1_196693659res_block_1_196693661res_block_1_196693663res_block_1_196693665res_block_1_196693667res_block_1_196693669res_block_1_196693671res_block_1_196693673res_block_1_196693675res_block_1_196693677res_block_1_196693679res_block_1_196693681res_block_1_196693683res_block_1_196693685*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_1_layer_call_and_return_conditional_losses_1966926082%
#res_block_1/StatefulPartitionedCall?
tf.__operators__.add_1/AddV2AddV2relu_0/PartitionedCall:output:0,res_block_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add_1/AddV2?
relu_1/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_relu_1_layer_call_and_return_conditional_losses_1966933882
relu_1/PartitionedCall?
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_196693690feature_196693692feature_196693694feature_196693696feature_196693698feature_196693700*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_feature_layer_call_and_return_conditional_losses_1966929922!
feature/StatefulPartitionedCall?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose(feature/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????R2"
 tf.compat.v1.transpose/transpose?
feature_linear/PartitionedCallPartitionedCall$tf.compat.v1.transpose/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_feature_linear_layer_call_and_return_conditional_losses_1966934492 
feature_linear/PartitionedCall?
IdentityIdentity'feature_linear/PartitionedCall:output:0 ^feature/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^res_block_0/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
feature/StatefulPartitionedCallfeature/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#res_block_0/StatefulPartitionedCall#res_block_0/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_1_layer_call_fn_196695776

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1966908272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?V
?
J__inference_res_block_0_layer_call_and_return_conditional_losses_196695071

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource
identity??5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_2/BiasAdd?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3?
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_2/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dactivation_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_3/BiasAdd?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3?
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_3/Relu?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_4/BiasAdd?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3?
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_3_layer_call_fn_196696082

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1966912272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_8_layer_call_fn_196696814

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1966927942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?/
?
J__inference_res_block_1_layer_call_and_return_conditional_losses_196692557
conv2d_5_input
conv2d_5_196692512
conv2d_5_196692514#
batch_normalization_5_196692517#
batch_normalization_5_196692519#
batch_normalization_5_196692521#
batch_normalization_5_196692523
conv2d_6_196692527
conv2d_6_196692529#
batch_normalization_6_196692532#
batch_normalization_6_196692534#
batch_normalization_6_196692536#
batch_normalization_6_196692538
conv2d_7_196692542
conv2d_7_196692544#
batch_normalization_7_196692547#
batch_normalization_7_196692549#
batch_normalization_7_196692551#
batch_normalization_7_196692553
identity??-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_196692512conv2d_5_196692514*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_1966922012"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_196692517batch_normalization_5_196692519batch_normalization_5_196692521batch_normalization_5_196692523*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1966922522/
-batch_normalization_5/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_4_layer_call_and_return_conditional_losses_1966922932
activation_4/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_196692527conv2d_6_196692529*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_1966923112"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_196692532batch_normalization_6_196692534batch_normalization_6_196692536batch_normalization_6_196692538*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1966923622/
-batch_normalization_6/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_5_layer_call_and_return_conditional_losses_1966924032
activation_5/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_196692542conv2d_7_196692544*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_7_layer_call_and_return_conditional_losses_1966924212"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_196692547batch_normalization_7_196692549batch_normalization_7_196692551batch_normalization_7_196692553*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1966924722/
-batch_normalization_7/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_7/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_5_input
?	
?
G__inference_conv2d_1_layer_call_and_return_conditional_losses_196690776

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196690541

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
L
0__inference_activation_5_layer_call_fn_196696541

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_5_layer_call_and_return_conditional_losses_1966924032
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
a
E__inference_relu_0_layer_call_and_return_conditional_losses_196693255

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
F__inference_feature_layer_call_and_return_conditional_losses_196692992

inputs
conv2d_8_196692976
conv2d_8_196692978#
batch_normalization_8_196692981#
batch_normalization_8_196692983#
batch_normalization_8_196692985#
batch_normalization_8_196692987
identity??-batch_normalization_8/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_196692976conv2d_8_196692978*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_8_layer_call_and_return_conditional_losses_1966928502"
 conv2d_8/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_196692981batch_normalization_8_196692983batch_normalization_8_196692985batch_normalization_8_196692987*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1966928832/
-batch_normalization_8/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_6_layer_call_and_return_conditional_losses_1966929422
activation_6/PartitionedCall?
IdentityIdentity%activation_6/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_4_layer_call_and_return_conditional_losses_196691572

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_7_layer_call_and_return_conditional_losses_196692421

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_5_layer_call_fn_196696378

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1966922522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196696640

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
F__inference_feature_layer_call_and_return_conditional_losses_196692970
conv2d_8_input
conv2d_8_196692954
conv2d_8_196692956#
batch_normalization_8_196692959#
batch_normalization_8_196692961#
batch_normalization_8_196692963#
batch_normalization_8_196692965
identity??-batch_normalization_8/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_196692954conv2d_8_196692956*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_8_layer_call_and_return_conditional_losses_1966928502"
 conv2d_8/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_196692959batch_normalization_8_196692961batch_normalization_8_196692963batch_normalization_8_196692965*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1966929012/
-batch_normalization_8/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_6_layer_call_and_return_conditional_losses_1966929422
activation_6/PartitionedCall?
IdentityIdentity%activation_6/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_8_input
?
?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196691127

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_3_layer_call_and_return_conditional_losses_196695949

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
g
K__inference_activation_4_layer_call_and_return_conditional_losses_196692293

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196691327

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196691513

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196696290

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196692252

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_7_layer_call_fn_196696609

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1966921452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?V
?
J__inference_res_block_1_layer_call_and_return_conditional_losses_196695295

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource
identity??5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_5/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3?
activation_4/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_4/Relu?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dactivation_4/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_6/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3?
activation_5/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_5/Relu?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Dactivation_5/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_7/BiasAdd?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3?
IdentityIdentity*batch_normalization_7/FusedBatchNormV3:y:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_3_layer_call_fn_196696007

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1966914952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
g
K__inference_activation_6_layer_call_and_return_conditional_losses_196696832

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_6_layer_call_fn_196696518

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1966920452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
)__inference_model_layer_call_fn_196694789

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_1966939432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196696801

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196696578

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
+__inference_feature_layer_call_fn_196693007
conv2d_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_feature_layer_call_and_return_conditional_losses_1966929922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_8_input
?	
?
G__inference_conv2d_2_layer_call_and_return_conditional_losses_196691352

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_7_layer_call_fn_196696684

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1966924722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
,__inference_conv2d_7_layer_call_fn_196696560

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
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_7_layer_call_and_return_conditional_losses_1966924212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196695976

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
g
K__inference_activation_5_layer_call_and_return_conditional_losses_196696536

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_3_layer_call_fn_196696020

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1966915132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
a
E__inference_relu_0_layer_call_and_return_conditional_losses_196695158

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
g
K__inference_activation_2_layer_call_and_return_conditional_losses_196695934

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
L
0__inference_activation_4_layer_call_fn_196696388

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_4_layer_call_and_return_conditional_losses_1966922932
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
)__inference_model_layer_call_fn_196694676

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_1966937072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_layer_call_fn_196695623

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1966905412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196691495

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_1_layer_call_fn_196695714

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1966906412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?8
$__inference__wrapped_model_196690452	
input:
6model_input_conv_conv2d_conv2d_readvariableop_resource;
7model_input_conv_conv2d_biasadd_readvariableop_resource@
<model_input_conv_batch_normalization_readvariableop_resourceB
>model_input_conv_batch_normalization_readvariableop_1_resourceQ
Mmodel_input_conv_batch_normalization_fusedbatchnormv3_readvariableop_resourceS
Omodel_input_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource<
8model_input_conv_conv2d_1_conv2d_readvariableop_resource=
9model_input_conv_conv2d_1_biasadd_readvariableop_resourceB
>model_input_conv_batch_normalization_1_readvariableop_resourceD
@model_input_conv_batch_normalization_1_readvariableop_1_resourceS
Omodel_input_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceU
Qmodel_input_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource=
9model_res_block_0_conv2d_2_conv2d_readvariableop_resource>
:model_res_block_0_conv2d_2_biasadd_readvariableop_resourceC
?model_res_block_0_batch_normalization_2_readvariableop_resourceE
Amodel_res_block_0_batch_normalization_2_readvariableop_1_resourceT
Pmodel_res_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceV
Rmodel_res_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource=
9model_res_block_0_conv2d_3_conv2d_readvariableop_resource>
:model_res_block_0_conv2d_3_biasadd_readvariableop_resourceC
?model_res_block_0_batch_normalization_3_readvariableop_resourceE
Amodel_res_block_0_batch_normalization_3_readvariableop_1_resourceT
Pmodel_res_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceV
Rmodel_res_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource=
9model_res_block_0_conv2d_4_conv2d_readvariableop_resource>
:model_res_block_0_conv2d_4_biasadd_readvariableop_resourceC
?model_res_block_0_batch_normalization_4_readvariableop_resourceE
Amodel_res_block_0_batch_normalization_4_readvariableop_1_resourceT
Pmodel_res_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceV
Rmodel_res_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource=
9model_res_block_1_conv2d_5_conv2d_readvariableop_resource>
:model_res_block_1_conv2d_5_biasadd_readvariableop_resourceC
?model_res_block_1_batch_normalization_5_readvariableop_resourceE
Amodel_res_block_1_batch_normalization_5_readvariableop_1_resourceT
Pmodel_res_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceV
Rmodel_res_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource=
9model_res_block_1_conv2d_6_conv2d_readvariableop_resource>
:model_res_block_1_conv2d_6_biasadd_readvariableop_resourceC
?model_res_block_1_batch_normalization_6_readvariableop_resourceE
Amodel_res_block_1_batch_normalization_6_readvariableop_1_resourceT
Pmodel_res_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceV
Rmodel_res_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource=
9model_res_block_1_conv2d_7_conv2d_readvariableop_resource>
:model_res_block_1_conv2d_7_biasadd_readvariableop_resourceC
?model_res_block_1_batch_normalization_7_readvariableop_resourceE
Amodel_res_block_1_batch_normalization_7_readvariableop_1_resourceT
Pmodel_res_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceV
Rmodel_res_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource9
5model_feature_conv2d_8_conv2d_readvariableop_resource:
6model_feature_conv2d_8_biasadd_readvariableop_resource?
;model_feature_batch_normalization_8_readvariableop_resourceA
=model_feature_batch_normalization_8_readvariableop_1_resourceP
Lmodel_feature_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceR
Nmodel_feature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??Cmodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?Emodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?2model/feature/batch_normalization_8/ReadVariableOp?4model/feature/batch_normalization_8/ReadVariableOp_1?-model/feature/conv2d_8/BiasAdd/ReadVariableOp?,model/feature/conv2d_8/Conv2D/ReadVariableOp?Dmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp?Fmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?3model/input_conv/batch_normalization/ReadVariableOp?5model/input_conv/batch_normalization/ReadVariableOp_1?Fmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Hmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?5model/input_conv/batch_normalization_1/ReadVariableOp?7model/input_conv/batch_normalization_1/ReadVariableOp_1?.model/input_conv/conv2d/BiasAdd/ReadVariableOp?-model/input_conv/conv2d/Conv2D/ReadVariableOp?0model/input_conv/conv2d_1/BiasAdd/ReadVariableOp?/model/input_conv/conv2d_1/Conv2D/ReadVariableOp?Gmodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?Imodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?6model/res_block_0/batch_normalization_2/ReadVariableOp?8model/res_block_0/batch_normalization_2/ReadVariableOp_1?Gmodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Imodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?6model/res_block_0/batch_normalization_3/ReadVariableOp?8model/res_block_0/batch_normalization_3/ReadVariableOp_1?Gmodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Imodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?6model/res_block_0/batch_normalization_4/ReadVariableOp?8model/res_block_0/batch_normalization_4/ReadVariableOp_1?1model/res_block_0/conv2d_2/BiasAdd/ReadVariableOp?0model/res_block_0/conv2d_2/Conv2D/ReadVariableOp?1model/res_block_0/conv2d_3/BiasAdd/ReadVariableOp?0model/res_block_0/conv2d_3/Conv2D/ReadVariableOp?1model/res_block_0/conv2d_4/BiasAdd/ReadVariableOp?0model/res_block_0/conv2d_4/Conv2D/ReadVariableOp?Gmodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Imodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?6model/res_block_1/batch_normalization_5/ReadVariableOp?8model/res_block_1/batch_normalization_5/ReadVariableOp_1?Gmodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Imodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?6model/res_block_1/batch_normalization_6/ReadVariableOp?8model/res_block_1/batch_normalization_6/ReadVariableOp_1?Gmodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Imodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?6model/res_block_1/batch_normalization_7/ReadVariableOp?8model/res_block_1/batch_normalization_7/ReadVariableOp_1?1model/res_block_1/conv2d_5/BiasAdd/ReadVariableOp?0model/res_block_1/conv2d_5/Conv2D/ReadVariableOp?1model/res_block_1/conv2d_6/BiasAdd/ReadVariableOp?0model/res_block_1/conv2d_6/Conv2D/ReadVariableOp?1model/res_block_1/conv2d_7/BiasAdd/ReadVariableOp?0model/res_block_1/conv2d_7/Conv2D/ReadVariableOp?
-model/input_conv/conv2d/Conv2D/ReadVariableOpReadVariableOp6model_input_conv_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02/
-model/input_conv/conv2d/Conv2D/ReadVariableOp?
model/input_conv/conv2d/Conv2DConv2Dinput5model/input_conv/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2 
model/input_conv/conv2d/Conv2D?
.model/input_conv/conv2d/BiasAdd/ReadVariableOpReadVariableOp7model_input_conv_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.model/input_conv/conv2d/BiasAdd/ReadVariableOp?
model/input_conv/conv2d/BiasAddBiasAdd'model/input_conv/conv2d/Conv2D:output:06model/input_conv/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2!
model/input_conv/conv2d/BiasAdd?
3model/input_conv/batch_normalization/ReadVariableOpReadVariableOp<model_input_conv_batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype025
3model/input_conv/batch_normalization/ReadVariableOp?
5model/input_conv/batch_normalization/ReadVariableOp_1ReadVariableOp>model_input_conv_batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype027
5model/input_conv/batch_normalization/ReadVariableOp_1?
Dmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMmodel_input_conv_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02F
Dmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Fmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOmodel_input_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02H
Fmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
5model/input_conv/batch_normalization/FusedBatchNormV3FusedBatchNormV3(model/input_conv/conv2d/BiasAdd:output:0;model/input_conv/batch_normalization/ReadVariableOp:value:0=model/input_conv/batch_normalization/ReadVariableOp_1:value:0Lmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Nmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 27
5model/input_conv/batch_normalization/FusedBatchNormV3?
 model/input_conv/activation/ReluRelu9model/input_conv/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2"
 model/input_conv/activation/Relu?
/model/input_conv/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8model_input_conv_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype021
/model/input_conv/conv2d_1/Conv2D/ReadVariableOp?
 model/input_conv/conv2d_1/Conv2DConv2D.model/input_conv/activation/Relu:activations:07model/input_conv/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingVALID*
strides
2"
 model/input_conv/conv2d_1/Conv2D?
0model/input_conv/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9model_input_conv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0model/input_conv/conv2d_1/BiasAdd/ReadVariableOp?
!model/input_conv/conv2d_1/BiasAddBiasAdd)model/input_conv/conv2d_1/Conv2D:output:08model/input_conv/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2#
!model/input_conv/conv2d_1/BiasAdd?
5model/input_conv/batch_normalization_1/ReadVariableOpReadVariableOp>model_input_conv_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype027
5model/input_conv/batch_normalization_1/ReadVariableOp?
7model/input_conv/batch_normalization_1/ReadVariableOp_1ReadVariableOp@model_input_conv_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7model/input_conv/batch_normalization_1/ReadVariableOp_1?
Fmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodel_input_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02H
Fmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Hmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodel_input_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02J
Hmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
7model/input_conv/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3*model/input_conv/conv2d_1/BiasAdd:output:0=model/input_conv/batch_normalization_1/ReadVariableOp:value:0?model/input_conv/batch_normalization_1/ReadVariableOp_1:value:0Nmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Pmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 29
7model/input_conv/batch_normalization_1/FusedBatchNormV3?
"model/input_conv/activation_1/ReluRelu;model/input_conv/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2$
"model/input_conv/activation_1/Relu?
0model/res_block_0/conv2d_2/Conv2D/ReadVariableOpReadVariableOp9model_res_block_0_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/res_block_0/conv2d_2/Conv2D/ReadVariableOp?
!model/res_block_0/conv2d_2/Conv2DConv2D0model/input_conv/activation_1/Relu:activations:08model/res_block_0/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2#
!model/res_block_0/conv2d_2/Conv2D?
1model/res_block_0/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp:model_res_block_0_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/res_block_0/conv2d_2/BiasAdd/ReadVariableOp?
"model/res_block_0/conv2d_2/BiasAddBiasAdd*model/res_block_0/conv2d_2/Conv2D:output:09model/res_block_0/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2$
"model/res_block_0/conv2d_2/BiasAdd?
6model/res_block_0/batch_normalization_2/ReadVariableOpReadVariableOp?model_res_block_0_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/res_block_0/batch_normalization_2/ReadVariableOp?
8model/res_block_0/batch_normalization_2/ReadVariableOp_1ReadVariableOpAmodel_res_block_0_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8model/res_block_0/batch_normalization_2/ReadVariableOp_1?
Gmodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_res_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
Imodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_res_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
8model/res_block_0/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3+model/res_block_0/conv2d_2/BiasAdd:output:0>model/res_block_0/batch_normalization_2/ReadVariableOp:value:0@model/res_block_0/batch_normalization_2/ReadVariableOp_1:value:0Omodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2:
8model/res_block_0/batch_normalization_2/FusedBatchNormV3?
#model/res_block_0/activation_2/ReluRelu<model/res_block_0/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2%
#model/res_block_0/activation_2/Relu?
0model/res_block_0/conv2d_3/Conv2D/ReadVariableOpReadVariableOp9model_res_block_0_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/res_block_0/conv2d_3/Conv2D/ReadVariableOp?
!model/res_block_0/conv2d_3/Conv2DConv2D1model/res_block_0/activation_2/Relu:activations:08model/res_block_0/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2#
!model/res_block_0/conv2d_3/Conv2D?
1model/res_block_0/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp:model_res_block_0_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/res_block_0/conv2d_3/BiasAdd/ReadVariableOp?
"model/res_block_0/conv2d_3/BiasAddBiasAdd*model/res_block_0/conv2d_3/Conv2D:output:09model/res_block_0/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2$
"model/res_block_0/conv2d_3/BiasAdd?
6model/res_block_0/batch_normalization_3/ReadVariableOpReadVariableOp?model_res_block_0_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/res_block_0/batch_normalization_3/ReadVariableOp?
8model/res_block_0/batch_normalization_3/ReadVariableOp_1ReadVariableOpAmodel_res_block_0_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8model/res_block_0/batch_normalization_3/ReadVariableOp_1?
Gmodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_res_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Imodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_res_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
8model/res_block_0/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3+model/res_block_0/conv2d_3/BiasAdd:output:0>model/res_block_0/batch_normalization_3/ReadVariableOp:value:0@model/res_block_0/batch_normalization_3/ReadVariableOp_1:value:0Omodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2:
8model/res_block_0/batch_normalization_3/FusedBatchNormV3?
#model/res_block_0/activation_3/ReluRelu<model/res_block_0/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2%
#model/res_block_0/activation_3/Relu?
0model/res_block_0/conv2d_4/Conv2D/ReadVariableOpReadVariableOp9model_res_block_0_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/res_block_0/conv2d_4/Conv2D/ReadVariableOp?
!model/res_block_0/conv2d_4/Conv2DConv2D1model/res_block_0/activation_3/Relu:activations:08model/res_block_0/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2#
!model/res_block_0/conv2d_4/Conv2D?
1model/res_block_0/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp:model_res_block_0_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/res_block_0/conv2d_4/BiasAdd/ReadVariableOp?
"model/res_block_0/conv2d_4/BiasAddBiasAdd*model/res_block_0/conv2d_4/Conv2D:output:09model/res_block_0/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2$
"model/res_block_0/conv2d_4/BiasAdd?
6model/res_block_0/batch_normalization_4/ReadVariableOpReadVariableOp?model_res_block_0_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/res_block_0/batch_normalization_4/ReadVariableOp?
8model/res_block_0/batch_normalization_4/ReadVariableOp_1ReadVariableOpAmodel_res_block_0_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8model/res_block_0/batch_normalization_4/ReadVariableOp_1?
Gmodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_res_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Imodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_res_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
8model/res_block_0/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3+model/res_block_0/conv2d_4/BiasAdd:output:0>model/res_block_0/batch_normalization_4/ReadVariableOp:value:0@model/res_block_0/batch_normalization_4/ReadVariableOp_1:value:0Omodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2:
8model/res_block_0/batch_normalization_4/FusedBatchNormV3?
 model/tf.__operators__.add/AddV2AddV20model/input_conv/activation_1/Relu:activations:0<model/res_block_0/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2"
 model/tf.__operators__.add/AddV2?
model/relu_0/ReluRelu$model/tf.__operators__.add/AddV2:z:0*
T0*/
_output_shapes
:?????????R@2
model/relu_0/Relu?
0model/res_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp9model_res_block_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/res_block_1/conv2d_5/Conv2D/ReadVariableOp?
!model/res_block_1/conv2d_5/Conv2DConv2Dmodel/relu_0/Relu:activations:08model/res_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2#
!model/res_block_1/conv2d_5/Conv2D?
1model/res_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp:model_res_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/res_block_1/conv2d_5/BiasAdd/ReadVariableOp?
"model/res_block_1/conv2d_5/BiasAddBiasAdd*model/res_block_1/conv2d_5/Conv2D:output:09model/res_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2$
"model/res_block_1/conv2d_5/BiasAdd?
6model/res_block_1/batch_normalization_5/ReadVariableOpReadVariableOp?model_res_block_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/res_block_1/batch_normalization_5/ReadVariableOp?
8model/res_block_1/batch_normalization_5/ReadVariableOp_1ReadVariableOpAmodel_res_block_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8model/res_block_1/batch_normalization_5/ReadVariableOp_1?
Gmodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_res_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Imodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_res_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
8model/res_block_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3+model/res_block_1/conv2d_5/BiasAdd:output:0>model/res_block_1/batch_normalization_5/ReadVariableOp:value:0@model/res_block_1/batch_normalization_5/ReadVariableOp_1:value:0Omodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2:
8model/res_block_1/batch_normalization_5/FusedBatchNormV3?
#model/res_block_1/activation_4/ReluRelu<model/res_block_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2%
#model/res_block_1/activation_4/Relu?
0model/res_block_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp9model_res_block_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/res_block_1/conv2d_6/Conv2D/ReadVariableOp?
!model/res_block_1/conv2d_6/Conv2DConv2D1model/res_block_1/activation_4/Relu:activations:08model/res_block_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2#
!model/res_block_1/conv2d_6/Conv2D?
1model/res_block_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp:model_res_block_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/res_block_1/conv2d_6/BiasAdd/ReadVariableOp?
"model/res_block_1/conv2d_6/BiasAddBiasAdd*model/res_block_1/conv2d_6/Conv2D:output:09model/res_block_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2$
"model/res_block_1/conv2d_6/BiasAdd?
6model/res_block_1/batch_normalization_6/ReadVariableOpReadVariableOp?model_res_block_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/res_block_1/batch_normalization_6/ReadVariableOp?
8model/res_block_1/batch_normalization_6/ReadVariableOp_1ReadVariableOpAmodel_res_block_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8model/res_block_1/batch_normalization_6/ReadVariableOp_1?
Gmodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_res_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Imodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_res_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
8model/res_block_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3+model/res_block_1/conv2d_6/BiasAdd:output:0>model/res_block_1/batch_normalization_6/ReadVariableOp:value:0@model/res_block_1/batch_normalization_6/ReadVariableOp_1:value:0Omodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2:
8model/res_block_1/batch_normalization_6/FusedBatchNormV3?
#model/res_block_1/activation_5/ReluRelu<model/res_block_1/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2%
#model/res_block_1/activation_5/Relu?
0model/res_block_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp9model_res_block_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/res_block_1/conv2d_7/Conv2D/ReadVariableOp?
!model/res_block_1/conv2d_7/Conv2DConv2D1model/res_block_1/activation_5/Relu:activations:08model/res_block_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2#
!model/res_block_1/conv2d_7/Conv2D?
1model/res_block_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp:model_res_block_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model/res_block_1/conv2d_7/BiasAdd/ReadVariableOp?
"model/res_block_1/conv2d_7/BiasAddBiasAdd*model/res_block_1/conv2d_7/Conv2D:output:09model/res_block_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2$
"model/res_block_1/conv2d_7/BiasAdd?
6model/res_block_1/batch_normalization_7/ReadVariableOpReadVariableOp?model_res_block_1_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype028
6model/res_block_1/batch_normalization_7/ReadVariableOp?
8model/res_block_1/batch_normalization_7/ReadVariableOp_1ReadVariableOpAmodel_res_block_1_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8model/res_block_1/batch_normalization_7/ReadVariableOp_1?
Gmodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpPmodel_res_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Imodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRmodel_res_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
8model/res_block_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3+model/res_block_1/conv2d_7/BiasAdd:output:0>model/res_block_1/batch_normalization_7/ReadVariableOp:value:0@model/res_block_1/batch_normalization_7/ReadVariableOp_1:value:0Omodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Qmodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2:
8model/res_block_1/batch_normalization_7/FusedBatchNormV3?
"model/tf.__operators__.add_1/AddV2AddV2model/relu_0/Relu:activations:0<model/res_block_1/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2$
"model/tf.__operators__.add_1/AddV2?
model/relu_1/ReluRelu&model/tf.__operators__.add_1/AddV2:z:0*
T0*/
_output_shapes
:?????????R@2
model/relu_1/Relu?
,model/feature/conv2d_8/Conv2D/ReadVariableOpReadVariableOp5model_feature_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02.
,model/feature/conv2d_8/Conv2D/ReadVariableOp?
model/feature/conv2d_8/Conv2DConv2Dmodel/relu_1/Relu:activations:04model/feature/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R*
paddingVALID*
strides
2
model/feature/conv2d_8/Conv2D?
-model/feature/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp6model_feature_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-model/feature/conv2d_8/BiasAdd/ReadVariableOp?
model/feature/conv2d_8/BiasAddBiasAdd&model/feature/conv2d_8/Conv2D:output:05model/feature/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R2 
model/feature/conv2d_8/BiasAdd?
2model/feature/batch_normalization_8/ReadVariableOpReadVariableOp;model_feature_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype024
2model/feature/batch_normalization_8/ReadVariableOp?
4model/feature/batch_normalization_8/ReadVariableOp_1ReadVariableOp=model_feature_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype026
4model/feature/batch_normalization_8/ReadVariableOp_1?
Cmodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpLmodel_feature_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cmodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Emodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNmodel_feature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02G
Emodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
4model/feature/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3'model/feature/conv2d_8/BiasAdd:output:0:model/feature/batch_normalization_8/ReadVariableOp:value:0<model/feature/batch_normalization_8/ReadVariableOp_1:value:0Kmodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Mmodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 26
4model/feature/batch_normalization_8/FusedBatchNormV3?
model/feature/activation_6/ReluRelu8model/feature/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R2!
model/feature/activation_6/Relu?
+model/tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+model/tf.compat.v1.transpose/transpose/perm?
&model/tf.compat.v1.transpose/transpose	Transpose-model/feature/activation_6/Relu:activations:04model/tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????R2(
&model/tf.compat.v1.transpose/transpose?
IdentityIdentity*model/tf.compat.v1.transpose/transpose:y:0D^model/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpF^model/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_13^model/feature/batch_normalization_8/ReadVariableOp5^model/feature/batch_normalization_8/ReadVariableOp_1.^model/feature/conv2d_8/BiasAdd/ReadVariableOp-^model/feature/conv2d_8/Conv2D/ReadVariableOpE^model/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpG^model/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_14^model/input_conv/batch_normalization/ReadVariableOp6^model/input_conv/batch_normalization/ReadVariableOp_1G^model/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpI^model/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_16^model/input_conv/batch_normalization_1/ReadVariableOp8^model/input_conv/batch_normalization_1/ReadVariableOp_1/^model/input_conv/conv2d/BiasAdd/ReadVariableOp.^model/input_conv/conv2d/Conv2D/ReadVariableOp1^model/input_conv/conv2d_1/BiasAdd/ReadVariableOp0^model/input_conv/conv2d_1/Conv2D/ReadVariableOpH^model/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpJ^model/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17^model/res_block_0/batch_normalization_2/ReadVariableOp9^model/res_block_0/batch_normalization_2/ReadVariableOp_1H^model/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpJ^model/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17^model/res_block_0/batch_normalization_3/ReadVariableOp9^model/res_block_0/batch_normalization_3/ReadVariableOp_1H^model/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpJ^model/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17^model/res_block_0/batch_normalization_4/ReadVariableOp9^model/res_block_0/batch_normalization_4/ReadVariableOp_12^model/res_block_0/conv2d_2/BiasAdd/ReadVariableOp1^model/res_block_0/conv2d_2/Conv2D/ReadVariableOp2^model/res_block_0/conv2d_3/BiasAdd/ReadVariableOp1^model/res_block_0/conv2d_3/Conv2D/ReadVariableOp2^model/res_block_0/conv2d_4/BiasAdd/ReadVariableOp1^model/res_block_0/conv2d_4/Conv2D/ReadVariableOpH^model/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpJ^model/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17^model/res_block_1/batch_normalization_5/ReadVariableOp9^model/res_block_1/batch_normalization_5/ReadVariableOp_1H^model/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpJ^model/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17^model/res_block_1/batch_normalization_6/ReadVariableOp9^model/res_block_1/batch_normalization_6/ReadVariableOp_1H^model/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpJ^model/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17^model/res_block_1/batch_normalization_7/ReadVariableOp9^model/res_block_1/batch_normalization_7/ReadVariableOp_12^model/res_block_1/conv2d_5/BiasAdd/ReadVariableOp1^model/res_block_1/conv2d_5/Conv2D/ReadVariableOp2^model/res_block_1/conv2d_6/BiasAdd/ReadVariableOp1^model/res_block_1/conv2d_6/Conv2D/ReadVariableOp2^model/res_block_1/conv2d_7/BiasAdd/ReadVariableOp1^model/res_block_1/conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::2?
Cmodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpCmodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
Emodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Emodel/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12h
2model/feature/batch_normalization_8/ReadVariableOp2model/feature/batch_normalization_8/ReadVariableOp2l
4model/feature/batch_normalization_8/ReadVariableOp_14model/feature/batch_normalization_8/ReadVariableOp_12^
-model/feature/conv2d_8/BiasAdd/ReadVariableOp-model/feature/conv2d_8/BiasAdd/ReadVariableOp2\
,model/feature/conv2d_8/Conv2D/ReadVariableOp,model/feature/conv2d_8/Conv2D/ReadVariableOp2?
Dmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpDmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Fmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Fmodel/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3model/input_conv/batch_normalization/ReadVariableOp3model/input_conv/batch_normalization/ReadVariableOp2n
5model/input_conv/batch_normalization/ReadVariableOp_15model/input_conv/batch_normalization/ReadVariableOp_12?
Fmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpFmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Hmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Hmodel/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12n
5model/input_conv/batch_normalization_1/ReadVariableOp5model/input_conv/batch_normalization_1/ReadVariableOp2r
7model/input_conv/batch_normalization_1/ReadVariableOp_17model/input_conv/batch_normalization_1/ReadVariableOp_12`
.model/input_conv/conv2d/BiasAdd/ReadVariableOp.model/input_conv/conv2d/BiasAdd/ReadVariableOp2^
-model/input_conv/conv2d/Conv2D/ReadVariableOp-model/input_conv/conv2d/Conv2D/ReadVariableOp2d
0model/input_conv/conv2d_1/BiasAdd/ReadVariableOp0model/input_conv/conv2d_1/BiasAdd/ReadVariableOp2b
/model/input_conv/conv2d_1/Conv2D/ReadVariableOp/model/input_conv/conv2d_1/Conv2D/ReadVariableOp2?
Gmodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpGmodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2?
Imodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Imodel/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12p
6model/res_block_0/batch_normalization_2/ReadVariableOp6model/res_block_0/batch_normalization_2/ReadVariableOp2t
8model/res_block_0/batch_normalization_2/ReadVariableOp_18model/res_block_0/batch_normalization_2/ReadVariableOp_12?
Gmodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpGmodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Imodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Imodel/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12p
6model/res_block_0/batch_normalization_3/ReadVariableOp6model/res_block_0/batch_normalization_3/ReadVariableOp2t
8model/res_block_0/batch_normalization_3/ReadVariableOp_18model/res_block_0/batch_normalization_3/ReadVariableOp_12?
Gmodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpGmodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Imodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Imodel/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12p
6model/res_block_0/batch_normalization_4/ReadVariableOp6model/res_block_0/batch_normalization_4/ReadVariableOp2t
8model/res_block_0/batch_normalization_4/ReadVariableOp_18model/res_block_0/batch_normalization_4/ReadVariableOp_12f
1model/res_block_0/conv2d_2/BiasAdd/ReadVariableOp1model/res_block_0/conv2d_2/BiasAdd/ReadVariableOp2d
0model/res_block_0/conv2d_2/Conv2D/ReadVariableOp0model/res_block_0/conv2d_2/Conv2D/ReadVariableOp2f
1model/res_block_0/conv2d_3/BiasAdd/ReadVariableOp1model/res_block_0/conv2d_3/BiasAdd/ReadVariableOp2d
0model/res_block_0/conv2d_3/Conv2D/ReadVariableOp0model/res_block_0/conv2d_3/Conv2D/ReadVariableOp2f
1model/res_block_0/conv2d_4/BiasAdd/ReadVariableOp1model/res_block_0/conv2d_4/BiasAdd/ReadVariableOp2d
0model/res_block_0/conv2d_4/Conv2D/ReadVariableOp0model/res_block_0/conv2d_4/Conv2D/ReadVariableOp2?
Gmodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpGmodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Imodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Imodel/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12p
6model/res_block_1/batch_normalization_5/ReadVariableOp6model/res_block_1/batch_normalization_5/ReadVariableOp2t
8model/res_block_1/batch_normalization_5/ReadVariableOp_18model/res_block_1/batch_normalization_5/ReadVariableOp_12?
Gmodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpGmodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Imodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Imodel/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12p
6model/res_block_1/batch_normalization_6/ReadVariableOp6model/res_block_1/batch_normalization_6/ReadVariableOp2t
8model/res_block_1/batch_normalization_6/ReadVariableOp_18model/res_block_1/batch_normalization_6/ReadVariableOp_12?
Gmodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpGmodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Imodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Imodel/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12p
6model/res_block_1/batch_normalization_7/ReadVariableOp6model/res_block_1/batch_normalization_7/ReadVariableOp2t
8model/res_block_1/batch_normalization_7/ReadVariableOp_18model/res_block_1/batch_normalization_7/ReadVariableOp_12f
1model/res_block_1/conv2d_5/BiasAdd/ReadVariableOp1model/res_block_1/conv2d_5/BiasAdd/ReadVariableOp2d
0model/res_block_1/conv2d_5/Conv2D/ReadVariableOp0model/res_block_1/conv2d_5/Conv2D/ReadVariableOp2f
1model/res_block_1/conv2d_6/BiasAdd/ReadVariableOp1model/res_block_1/conv2d_6/BiasAdd/ReadVariableOp2d
0model/res_block_1/conv2d_6/Conv2D/ReadVariableOp0model/res_block_1/conv2d_6/Conv2D/ReadVariableOp2f
1model/res_block_1/conv2d_7/BiasAdd/ReadVariableOp1model/res_block_1/conv2d_7/BiasAdd/ReadVariableOp2d
0model/res_block_1/conv2d_7/Conv2D/ReadVariableOp0model/res_block_1/conv2d_7/Conv2D/ReadVariableOp:W S
0
_output_shapes
:??????????

_user_specified_nameinput
?	
?
.__inference_input_conv_layer_call_fn_196694910

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_input_conv_layer_call_and_return_conditional_losses_1966909482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_layer_call_fn_196695561

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1966907172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
F__inference_feature_layer_call_and_return_conditional_losses_196695412

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R*
paddingVALID*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R2
conv2d_8/BiasAdd?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3?
activation_6/ReluRelu*batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R2
activation_6/Relu?
IdentityIdentityactivation_6/Relu:activations:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1 ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
)__inference_model_layer_call_fn_196693818	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_1966937072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:??????????

_user_specified_nameinput
?
?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196696056

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?g
?
"__inference__traced_save_196697022
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
value?B?7B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *E
dtypes;
9272
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :?:?:?:?:?:?:?@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:-)
'
_output_shapes
:?@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@:  

_output_shapes
:@: !

_output_shapes
:@: "

_output_shapes
:@: #

_output_shapes
:@: $

_output_shapes
:@:,%(
&
_output_shapes
:@@: &

_output_shapes
:@: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@: *

_output_shapes
:@:,+(
&
_output_shapes
:@@: ,

_output_shapes
:@: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@: 0

_output_shapes
:@:,1(
&
_output_shapes
:@: 2

_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
::7

_output_shapes
: 
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196695750

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
g
K__inference_activation_6_layer_call_and_return_conditional_losses_196692942

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?:
?
D__inference_model_layer_call_and_return_conditional_losses_196693581	
input
input_conv_196693461
input_conv_196693463
input_conv_196693465
input_conv_196693467
input_conv_196693469
input_conv_196693471
input_conv_196693473
input_conv_196693475
input_conv_196693477
input_conv_196693479
input_conv_196693481
input_conv_196693483
res_block_0_196693486
res_block_0_196693488
res_block_0_196693490
res_block_0_196693492
res_block_0_196693494
res_block_0_196693496
res_block_0_196693498
res_block_0_196693500
res_block_0_196693502
res_block_0_196693504
res_block_0_196693506
res_block_0_196693508
res_block_0_196693510
res_block_0_196693512
res_block_0_196693514
res_block_0_196693516
res_block_0_196693518
res_block_0_196693520
res_block_1_196693525
res_block_1_196693527
res_block_1_196693529
res_block_1_196693531
res_block_1_196693533
res_block_1_196693535
res_block_1_196693537
res_block_1_196693539
res_block_1_196693541
res_block_1_196693543
res_block_1_196693545
res_block_1_196693547
res_block_1_196693549
res_block_1_196693551
res_block_1_196693553
res_block_1_196693555
res_block_1_196693557
res_block_1_196693559
feature_196693564
feature_196693566
feature_196693568
feature_196693570
feature_196693572
feature_196693574
identity??feature/StatefulPartitionedCall?"input_conv/StatefulPartitionedCall?#res_block_0/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputinput_conv_196693461input_conv_196693463input_conv_196693465input_conv_196693467input_conv_196693469input_conv_196693471input_conv_196693473input_conv_196693475input_conv_196693477input_conv_196693479input_conv_196693481input_conv_196693483*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_input_conv_layer_call_and_return_conditional_losses_1966910112$
"input_conv/StatefulPartitionedCall?
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_196693486res_block_0_196693488res_block_0_196693490res_block_0_196693492res_block_0_196693494res_block_0_196693496res_block_0_196693498res_block_0_196693500res_block_0_196693502res_block_0_196693504res_block_0_196693506res_block_0_196693508res_block_0_196693510res_block_0_196693512res_block_0_196693514res_block_0_196693516res_block_0_196693518res_block_0_196693520*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_0_layer_call_and_return_conditional_losses_1966918482%
#res_block_0/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV2+input_conv/StatefulPartitionedCall:output:0,res_block_0/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add/AddV2?
relu_0/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_relu_0_layer_call_and_return_conditional_losses_1966932552
relu_0/PartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_196693525res_block_1_196693527res_block_1_196693529res_block_1_196693531res_block_1_196693533res_block_1_196693535res_block_1_196693537res_block_1_196693539res_block_1_196693541res_block_1_196693543res_block_1_196693545res_block_1_196693547res_block_1_196693549res_block_1_196693551res_block_1_196693553res_block_1_196693555res_block_1_196693557res_block_1_196693559*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_1_layer_call_and_return_conditional_losses_1966926972%
#res_block_1/StatefulPartitionedCall?
tf.__operators__.add_1/AddV2AddV2relu_0/PartitionedCall:output:0,res_block_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add_1/AddV2?
relu_1/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_relu_1_layer_call_and_return_conditional_losses_1966933882
relu_1/PartitionedCall?
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_196693564feature_196693566feature_196693568feature_196693570feature_196693572feature_196693574*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_feature_layer_call_and_return_conditional_losses_1966930282!
feature/StatefulPartitionedCall?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose(feature/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????R2"
 tf.compat.v1.transpose/transpose?
feature_linear/PartitionedCallPartitionedCall$tf.compat.v1.transpose/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_feature_linear_layer_call_and_return_conditional_losses_1966934492 
feature_linear/PartitionedCall?
IdentityIdentity'feature_linear/PartitionedCall:output:0 ^feature/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^res_block_0/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
feature/StatefulPartitionedCallfeature/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#res_block_0/StatefulPartitionedCall#res_block_0/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:W S
0
_output_shapes
:??????????

_user_specified_nameinput
?
?
7__inference_batch_normalization_layer_call_fn_196695610

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1966905102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_feature_layer_call_and_return_conditional_losses_196692951
conv2d_8_input
conv2d_8_196692861
conv2d_8_196692863#
batch_normalization_8_196692928#
batch_normalization_8_196692930#
batch_normalization_8_196692932#
batch_normalization_8_196692934
identity??-batch_normalization_8/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_196692861conv2d_8_196692863*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_8_layer_call_and_return_conditional_losses_1966928502"
 conv2d_8/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_196692928batch_normalization_8_196692930batch_normalization_8_196692932batch_normalization_8_196692934*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1966928832/
-batch_normalization_8/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_6_layer_call_and_return_conditional_losses_1966929422
activation_6/PartitionedCall?
IdentityIdentity%activation_6/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_8_input
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196690610

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196696272

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196696739

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_5_layer_call_fn_196696303

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1966919452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_conv2d_4_layer_call_fn_196696111

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
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_1966915722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196695579

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_7_layer_call_fn_196696671

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1966924542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196691227

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
/__inference_res_block_1_layer_call_fn_196695336

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_1_layer_call_and_return_conditional_losses_1966926082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196691403

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_5_layer_call_and_return_conditional_losses_196696245

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
g
K__inference_activation_1_layer_call_and_return_conditional_losses_196690868

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196692362

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196690641

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?/
?
J__inference_res_block_1_layer_call_and_return_conditional_losses_196692697

inputs
conv2d_5_196692652
conv2d_5_196692654#
batch_normalization_5_196692657#
batch_normalization_5_196692659#
batch_normalization_5_196692661#
batch_normalization_5_196692663
conv2d_6_196692667
conv2d_6_196692669#
batch_normalization_6_196692672#
batch_normalization_6_196692674#
batch_normalization_6_196692676#
batch_normalization_6_196692678
conv2d_7_196692682
conv2d_7_196692684#
batch_normalization_7_196692687#
batch_normalization_7_196692689#
batch_normalization_7_196692691#
batch_normalization_7_196692693
identity??-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_196692652conv2d_5_196692654*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_1966922012"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_196692657batch_normalization_5_196692659batch_normalization_5_196692661batch_normalization_5_196692663*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1966922522/
-batch_normalization_5/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_4_layer_call_and_return_conditional_losses_1966922932
activation_4/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_196692667conv2d_6_196692669*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_1966923112"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_196692672batch_normalization_6_196692674batch_normalization_6_196692676batch_normalization_6_196692678*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1966923622/
-batch_normalization_6/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_5_layer_call_and_return_conditional_losses_1966924032
activation_5/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_196692682conv2d_7_196692684*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_7_layer_call_and_return_conditional_losses_1966924212"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_196692687batch_normalization_7_196692689batch_normalization_7_196692691batch_normalization_7_196692693*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1966924722/
-batch_normalization_7/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_7/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196692176

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_2_layer_call_fn_196695929

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1966914032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196692145

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196696334

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_4_layer_call_and_return_conditional_losses_196696102

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196690809

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
+__inference_feature_layer_call_fn_196695471

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
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_feature_layer_call_and_return_conditional_losses_1966930282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196696783

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_2_layer_call_and_return_conditional_losses_196695796

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?/
?
J__inference_res_block_0_layer_call_and_return_conditional_losses_196691660
conv2d_2_input
conv2d_2_196691363
conv2d_2_196691365#
batch_normalization_2_196691430#
batch_normalization_2_196691432#
batch_normalization_2_196691434#
batch_normalization_2_196691436
conv2d_3_196691473
conv2d_3_196691475#
batch_normalization_3_196691540#
batch_normalization_3_196691542#
batch_normalization_3_196691544#
batch_normalization_3_196691546
conv2d_4_196691583
conv2d_4_196691585#
batch_normalization_4_196691650#
batch_normalization_4_196691652#
batch_normalization_4_196691654#
batch_normalization_4_196691656
identity??-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_196691363conv2d_2_196691365*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_1966913522"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_196691430batch_normalization_2_196691432batch_normalization_2_196691434batch_normalization_2_196691436*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1966913852/
-batch_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_2_layer_call_and_return_conditional_losses_1966914442
activation_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_196691473conv2d_3_196691475*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_1966914622"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_196691540batch_normalization_3_196691542batch_normalization_3_196691544batch_normalization_3_196691546*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1966914952/
-batch_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_3_layer_call_and_return_conditional_losses_1966915542
activation_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_196691583conv2d_4_196691585*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_1966915722"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_196691650batch_normalization_4_196691652batch_normalization_4_196691654batch_normalization_4_196691656*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1966916052/
-batch_normalization_4/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_2_input
?
g
K__inference_activation_5_layer_call_and_return_conditional_losses_196692403

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
L
0__inference_activation_2_layer_call_fn_196695939

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_2_layer_call_and_return_conditional_losses_1966914442
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_4_layer_call_fn_196696173

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1966913272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196696209

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196692234

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196691385

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
L
0__inference_activation_3_layer_call_fn_196696092

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_3_layer_call_and_return_conditional_losses_1966915542
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_5_layer_call_and_return_conditional_losses_196692201

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
.__inference_input_conv_layer_call_fn_196691038
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_input_conv_layer_call_and_return_conditional_losses_1966910112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:??????????
&
_user_specified_nameconv2d_input
?
?
9__inference_batch_normalization_1_layer_call_fn_196695763

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1966908092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
/__inference_res_block_0_layer_call_fn_196695153

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_0_layer_call_and_return_conditional_losses_1966918482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
'__inference_signature_wrapper_196694169	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__wrapped_model_1966904522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:??????????

_user_specified_nameinput
?
a
E__inference_relu_1_layer_call_and_return_conditional_losses_196693388

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196692344

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196690827

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196691945

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?/
?
J__inference_res_block_0_layer_call_and_return_conditional_losses_196691708
conv2d_2_input
conv2d_2_196691663
conv2d_2_196691665#
batch_normalization_2_196691668#
batch_normalization_2_196691670#
batch_normalization_2_196691672#
batch_normalization_2_196691674
conv2d_3_196691678
conv2d_3_196691680#
batch_normalization_3_196691683#
batch_normalization_3_196691685#
batch_normalization_3_196691687#
batch_normalization_3_196691689
conv2d_4_196691693
conv2d_4_196691695#
batch_normalization_4_196691698#
batch_normalization_4_196691700#
batch_normalization_4_196691702#
batch_normalization_4_196691704
identity??-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_196691663conv2d_2_196691665*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_1966913522"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_196691668batch_normalization_2_196691670batch_normalization_2_196691672batch_normalization_2_196691674*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1966914032/
-batch_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_2_layer_call_and_return_conditional_losses_1966914442
activation_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_196691678conv2d_3_196691680*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_1966914622"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_196691683batch_normalization_3_196691685batch_normalization_3_196691687batch_normalization_3_196691689*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1966915132/
-batch_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_3_layer_call_and_return_conditional_losses_1966915542
activation_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_196691693conv2d_4_196691695*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_1966915722"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_196691698batch_normalization_4_196691700batch_normalization_4_196691702batch_normalization_4_196691704*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1966916232/
-batch_normalization_4/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_2_input
?
?
9__inference_batch_normalization_3_layer_call_fn_196696069

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1966911962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_2_layer_call_fn_196695916

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1966913852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_6_layer_call_fn_196696456

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1966923442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196695517

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
E__inference_conv2d_layer_call_and_return_conditional_losses_196690666

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
i
M__inference_feature_linear_layer_call_and_return_conditional_losses_196695475

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_8_layer_call_and_return_conditional_losses_196696694

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
/__inference_res_block_1_layer_call_fn_196692647
conv2d_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_1_layer_call_and_return_conditional_losses_1966926082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_5_input
?/
?
J__inference_res_block_1_layer_call_and_return_conditional_losses_196692608

inputs
conv2d_5_196692563
conv2d_5_196692565#
batch_normalization_5_196692568#
batch_normalization_5_196692570#
batch_normalization_5_196692572#
batch_normalization_5_196692574
conv2d_6_196692578
conv2d_6_196692580#
batch_normalization_6_196692583#
batch_normalization_6_196692585#
batch_normalization_6_196692587#
batch_normalization_6_196692589
conv2d_7_196692593
conv2d_7_196692595#
batch_normalization_7_196692598#
batch_normalization_7_196692600#
batch_normalization_7_196692602#
batch_normalization_7_196692604
identity??-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_196692563conv2d_5_196692565*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_1966922012"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_196692568batch_normalization_5_196692570batch_normalization_5_196692572batch_normalization_5_196692574*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1966922342/
-batch_normalization_5/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_4_layer_call_and_return_conditional_losses_1966922932
activation_4/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_196692578conv2d_6_196692580*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_1966923112"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_196692583batch_normalization_6_196692585batch_normalization_6_196692587batch_normalization_6_196692589*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1966923442/
-batch_normalization_6/StatefulPartitionedCall?
activation_5/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_5_layer_call_and_return_conditional_losses_1966924032
activation_5/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_196692593conv2d_7_196692595*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_7_layer_call_and_return_conditional_losses_1966924212"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_196692598batch_normalization_7_196692600batch_normalization_7_196692602batch_normalization_7_196692604*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1966924542/
-batch_normalization_7/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_7/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
J
.__inference_activation_layer_call_fn_196695633

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_layer_call_and_return_conditional_losses_1966907582
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

*__inference_conv2d_layer_call_fn_196695499

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
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_1966906662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
.__inference_input_conv_layer_call_fn_196694939

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_input_conv_layer_call_and_return_conditional_losses_1966910112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196692794

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196696129

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196692454

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
,__inference_conv2d_6_layer_call_fn_196696407

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
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_6_layer_call_and_return_conditional_losses_1966923112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
i
M__inference_feature_linear_layer_call_and_return_conditional_losses_196693449

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196690717

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?:
?	
I__inference_input_conv_layer_call_and_return_conditional_losses_196694881

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
activation/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_1/Relu?
IdentityIdentityactivation_1/Relu:activations:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_model_layer_call_fn_196694054	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_1966939432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:??????????

_user_specified_nameinput
?
?
9__inference_batch_normalization_2_layer_call_fn_196695867

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1966911272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?:
?	
I__inference_input_conv_layer_call_and_return_conditional_losses_196694835

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
activation/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_1/BiasAdd?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_1/Relu?
IdentityIdentityactivation_1/Relu:activations:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196695732

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196692076

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_2_layer_call_fn_196695854

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1966910962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196696147

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196696487

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196696038

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196692045

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
/__inference_res_block_1_layer_call_fn_196695377

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_1_layer_call_and_return_conditional_losses_1966926972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
g
K__inference_activation_3_layer_call_and_return_conditional_losses_196691554

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
/__inference_res_block_0_layer_call_fn_196695112

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_0_layer_call_and_return_conditional_losses_1966917592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196691196

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?/
?
J__inference_res_block_0_layer_call_and_return_conditional_losses_196691759

inputs
conv2d_2_196691714
conv2d_2_196691716#
batch_normalization_2_196691719#
batch_normalization_2_196691721#
batch_normalization_2_196691723#
batch_normalization_2_196691725
conv2d_3_196691729
conv2d_3_196691731#
batch_normalization_3_196691734#
batch_normalization_3_196691736#
batch_normalization_3_196691738#
batch_normalization_3_196691740
conv2d_4_196691744
conv2d_4_196691746#
batch_normalization_4_196691749#
batch_normalization_4_196691751#
batch_normalization_4_196691753#
batch_normalization_4_196691755
identity??-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_196691714conv2d_2_196691716*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_1966913522"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_196691719batch_normalization_2_196691721batch_normalization_2_196691723batch_normalization_2_196691725*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1966913852/
-batch_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_2_layer_call_and_return_conditional_losses_1966914442
activation_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_196691729conv2d_3_196691731*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_1966914622"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_196691734batch_normalization_3_196691736batch_normalization_3_196691738batch_normalization_3_196691740*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1966914952/
-batch_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_3_layer_call_and_return_conditional_losses_1966915542
activation_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_196691744conv2d_4_196691746*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_1966915722"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_196691749batch_normalization_4_196691751batch_normalization_4_196691753batch_normalization_4_196691755*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1966916052/
-batch_normalization_4/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
+__inference_feature_layer_call_fn_196693043
conv2d_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_feature_layer_call_and_return_conditional_losses_1966930282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_8_input
?
a
E__inference_relu_1_layer_call_and_return_conditional_losses_196695382

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196696596

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_conv2d_1_layer_call_fn_196695652

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
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_1966907762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_6_layer_call_fn_196696531

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1966920762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_conv2d_8_layer_call_fn_196696703

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
:?????????R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_8_layer_call_and_return_conditional_losses_1966928502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_6_layer_call_and_return_conditional_losses_196696398

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196691623

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
N
2__inference_feature_linear_layer_call_fn_196695480

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_feature_linear_layer_call_and_return_conditional_losses_1966934492
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196695535

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
/__inference_res_block_0_layer_call_fn_196691887
conv2d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_0_layer_call_and_return_conditional_losses_1966918482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_2_input
?
?
9__inference_batch_normalization_8_layer_call_fn_196696765

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1966929012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
/__inference_res_block_0_layer_call_fn_196691798
conv2d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_0_layer_call_and_return_conditional_losses_1966917592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????R@
(
_user_specified_nameconv2d_2_input
?
F
*__inference_relu_1_layer_call_fn_196695387

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_relu_1_layer_call_and_return_conditional_losses_1966933882
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196696658

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_8_layer_call_and_return_conditional_losses_196692850

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
??
?
%__inference__traced_restore_196697194
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias0
,assignvariableop_2_batch_normalization_gamma/
+assignvariableop_3_batch_normalization_beta6
2assignvariableop_4_batch_normalization_moving_mean:
6assignvariableop_5_batch_normalization_moving_variance&
"assignvariableop_6_conv2d_1_kernel$
 assignvariableop_7_conv2d_1_bias2
.assignvariableop_8_batch_normalization_1_gamma1
-assignvariableop_9_batch_normalization_1_beta9
5assignvariableop_10_batch_normalization_1_moving_mean=
9assignvariableop_11_batch_normalization_1_moving_variance'
#assignvariableop_12_conv2d_2_kernel%
!assignvariableop_13_conv2d_2_bias3
/assignvariableop_14_batch_normalization_2_gamma2
.assignvariableop_15_batch_normalization_2_beta9
5assignvariableop_16_batch_normalization_2_moving_mean=
9assignvariableop_17_batch_normalization_2_moving_variance'
#assignvariableop_18_conv2d_3_kernel%
!assignvariableop_19_conv2d_3_bias3
/assignvariableop_20_batch_normalization_3_gamma2
.assignvariableop_21_batch_normalization_3_beta9
5assignvariableop_22_batch_normalization_3_moving_mean=
9assignvariableop_23_batch_normalization_3_moving_variance'
#assignvariableop_24_conv2d_4_kernel%
!assignvariableop_25_conv2d_4_bias3
/assignvariableop_26_batch_normalization_4_gamma2
.assignvariableop_27_batch_normalization_4_beta9
5assignvariableop_28_batch_normalization_4_moving_mean=
9assignvariableop_29_batch_normalization_4_moving_variance'
#assignvariableop_30_conv2d_5_kernel%
!assignvariableop_31_conv2d_5_bias3
/assignvariableop_32_batch_normalization_5_gamma2
.assignvariableop_33_batch_normalization_5_beta9
5assignvariableop_34_batch_normalization_5_moving_mean=
9assignvariableop_35_batch_normalization_5_moving_variance'
#assignvariableop_36_conv2d_6_kernel%
!assignvariableop_37_conv2d_6_bias3
/assignvariableop_38_batch_normalization_6_gamma2
.assignvariableop_39_batch_normalization_6_beta9
5assignvariableop_40_batch_normalization_6_moving_mean=
9assignvariableop_41_batch_normalization_6_moving_variance'
#assignvariableop_42_conv2d_7_kernel%
!assignvariableop_43_conv2d_7_bias3
/assignvariableop_44_batch_normalization_7_gamma2
.assignvariableop_45_batch_normalization_7_beta9
5assignvariableop_46_batch_normalization_7_moving_mean=
9assignvariableop_47_batch_normalization_7_moving_variance'
#assignvariableop_48_conv2d_8_kernel%
!assignvariableop_49_conv2d_8_bias3
/assignvariableop_50_batch_normalization_8_gamma2
.assignvariableop_51_batch_normalization_8_beta9
5assignvariableop_52_batch_normalization_8_moving_mean=
9assignvariableop_53_batch_normalization_8_moving_variance
identity_55??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
value?B?7B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
9272
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_4_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_4_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_5_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv2d_5_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_5_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_5_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_5_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_5_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp#assignvariableop_36_conv2d_6_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp!assignvariableop_37_conv2d_6_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp/assignvariableop_38_batch_normalization_6_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp.assignvariableop_39_batch_normalization_6_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp5assignvariableop_40_batch_normalization_6_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp9assignvariableop_41_batch_normalization_6_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp#assignvariableop_42_conv2d_7_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp!assignvariableop_43_conv2d_7_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp/assignvariableop_44_batch_normalization_7_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp.assignvariableop_45_batch_normalization_7_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp5assignvariableop_46_batch_normalization_7_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp9assignvariableop_47_batch_normalization_7_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp#assignvariableop_48_conv2d_8_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp!assignvariableop_49_conv2d_8_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp/assignvariableop_50_batch_normalization_8_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp.assignvariableop_51_batch_normalization_8_betaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp5assignvariableop_52_batch_normalization_8_moving_meanIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp9assignvariableop_53_batch_normalization_8_moving_varianceIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_539
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_54?	
Identity_55IdentityIdentity_54:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_55"#
identity_55Identity_55:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
9__inference_batch_normalization_6_layer_call_fn_196696469

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1966923622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196695688

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_5_layer_call_fn_196696365

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1966922342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196696425

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
g
K__inference_activation_4_layer_call_and_return_conditional_losses_196696383

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????R@:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?:
?
D__inference_model_layer_call_and_return_conditional_losses_196693458	
input
input_conv_196693105
input_conv_196693107
input_conv_196693109
input_conv_196693111
input_conv_196693113
input_conv_196693115
input_conv_196693117
input_conv_196693119
input_conv_196693121
input_conv_196693123
input_conv_196693125
input_conv_196693127
res_block_0_196693212
res_block_0_196693214
res_block_0_196693216
res_block_0_196693218
res_block_0_196693220
res_block_0_196693222
res_block_0_196693224
res_block_0_196693226
res_block_0_196693228
res_block_0_196693230
res_block_0_196693232
res_block_0_196693234
res_block_0_196693236
res_block_0_196693238
res_block_0_196693240
res_block_0_196693242
res_block_0_196693244
res_block_0_196693246
res_block_1_196693345
res_block_1_196693347
res_block_1_196693349
res_block_1_196693351
res_block_1_196693353
res_block_1_196693355
res_block_1_196693357
res_block_1_196693359
res_block_1_196693361
res_block_1_196693363
res_block_1_196693365
res_block_1_196693367
res_block_1_196693369
res_block_1_196693371
res_block_1_196693373
res_block_1_196693375
res_block_1_196693377
res_block_1_196693379
feature_196693430
feature_196693432
feature_196693434
feature_196693436
feature_196693438
feature_196693440
identity??feature/StatefulPartitionedCall?"input_conv/StatefulPartitionedCall?#res_block_0/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputinput_conv_196693105input_conv_196693107input_conv_196693109input_conv_196693111input_conv_196693113input_conv_196693115input_conv_196693117input_conv_196693119input_conv_196693121input_conv_196693123input_conv_196693125input_conv_196693127*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_input_conv_layer_call_and_return_conditional_losses_1966909482$
"input_conv/StatefulPartitionedCall?
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_196693212res_block_0_196693214res_block_0_196693216res_block_0_196693218res_block_0_196693220res_block_0_196693222res_block_0_196693224res_block_0_196693226res_block_0_196693228res_block_0_196693230res_block_0_196693232res_block_0_196693234res_block_0_196693236res_block_0_196693238res_block_0_196693240res_block_0_196693242res_block_0_196693244res_block_0_196693246*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_0_layer_call_and_return_conditional_losses_1966917592%
#res_block_0/StatefulPartitionedCall?
tf.__operators__.add/AddV2AddV2+input_conv/StatefulPartitionedCall:output:0,res_block_0/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add/AddV2?
relu_0/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_relu_0_layer_call_and_return_conditional_losses_1966932552
relu_0/PartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_196693345res_block_1_196693347res_block_1_196693349res_block_1_196693351res_block_1_196693353res_block_1_196693355res_block_1_196693357res_block_1_196693359res_block_1_196693361res_block_1_196693363res_block_1_196693365res_block_1_196693367res_block_1_196693369res_block_1_196693371res_block_1_196693373res_block_1_196693375res_block_1_196693377res_block_1_196693379*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_res_block_1_layer_call_and_return_conditional_losses_1966926082%
#res_block_1/StatefulPartitionedCall?
tf.__operators__.add_1/AddV2AddV2relu_0/PartitionedCall:output:0,res_block_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add_1/AddV2?
relu_1/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_relu_1_layer_call_and_return_conditional_losses_1966933882
relu_1/PartitionedCall?
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_196693430feature_196693432feature_196693434feature_196693436feature_196693438feature_196693440*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_feature_layer_call_and_return_conditional_losses_1966929922!
feature/StatefulPartitionedCall?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose(feature/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????R2"
 tf.compat.v1.transpose/transpose?
feature_linear/PartitionedCallPartitionedCall$tf.compat.v1.transpose/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_feature_linear_layer_call_and_return_conditional_losses_1966934492 
feature_linear/PartitionedCall?
IdentityIdentity'feature_linear/PartitionedCall:output:0 ^feature/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^res_block_0/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
feature/StatefulPartitionedCallfeature/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#res_block_0/StatefulPartitionedCall#res_block_0/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:W S
0
_output_shapes
:??????????

_user_specified_nameinput
?/
?
J__inference_res_block_0_layer_call_and_return_conditional_losses_196691848

inputs
conv2d_2_196691803
conv2d_2_196691805#
batch_normalization_2_196691808#
batch_normalization_2_196691810#
batch_normalization_2_196691812#
batch_normalization_2_196691814
conv2d_3_196691818
conv2d_3_196691820#
batch_normalization_3_196691823#
batch_normalization_3_196691825#
batch_normalization_3_196691827#
batch_normalization_3_196691829
conv2d_4_196691833
conv2d_4_196691835#
batch_normalization_4_196691838#
batch_normalization_4_196691840#
batch_normalization_4_196691842#
batch_normalization_4_196691844
identity??-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_196691803conv2d_2_196691805*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_1966913522"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_196691808batch_normalization_2_196691810batch_normalization_2_196691812batch_normalization_2_196691814*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1966914032/
-batch_normalization_2/StatefulPartitionedCall?
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_2_layer_call_and_return_conditional_losses_1966914442
activation_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_196691818conv2d_3_196691820*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_1966914622"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_196691823batch_normalization_3_196691825batch_normalization_3_196691827batch_normalization_3_196691829*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1966915132/
-batch_normalization_3/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_3_layer_call_and_return_conditional_losses_1966915542
activation_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_196691833conv2d_4_196691835*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_1966915722"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_196691838batch_normalization_4_196691840batch_normalization_4_196691842batch_normalization_4_196691844*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1966916232/
-batch_normalization_4/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_4_layer_call_fn_196696222

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1966916052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_4_layer_call_fn_196696235

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1966916232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196696505

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196692825

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_5_layer_call_fn_196696316

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1966919762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196690699

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
G__inference_conv2d_6_layer_call_and_return_conditional_losses_196692311

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?!
?
I__inference_input_conv_layer_call_and_return_conditional_losses_196690948

inputs
conv2d_196690917
conv2d_196690919!
batch_normalization_196690922!
batch_normalization_196690924!
batch_normalization_196690926!
batch_normalization_196690928
conv2d_1_196690932
conv2d_1_196690934#
batch_normalization_1_196690937#
batch_normalization_1_196690939#
batch_normalization_1_196690941#
batch_normalization_1_196690943
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_196690917conv2d_196690919*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_1966906662 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_196690922batch_normalization_196690924batch_normalization_196690926batch_normalization_196690928*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1966906992-
+batch_normalization/StatefulPartitionedCall?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_layer_call_and_return_conditional_losses_1966907582
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_196690932conv2d_1_196690934*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_1966907762"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_196690937batch_normalization_1_196690939batch_normalization_1_196690941batch_normalization_1_196690943*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1966908092/
-batch_normalization_1/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_1_layer_call_and_return_conditional_losses_1966908682
activation_1/PartitionedCall?
IdentityIdentity%activation_1/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:??????????::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196692883

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_layer_call_fn_196695548

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1966906992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196695885

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196695670

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196696721

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196692472

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?V
?
J__inference_res_block_1_layer_call_and_return_conditional_losses_196695229

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource
identity??5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_5/BiasAdd?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3?
activation_4/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_4/Relu?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dactivation_4/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_6/BiasAdd?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3?
activation_5/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
activation_5/Relu?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Dactivation_5/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
conv2d_7/BiasAdd?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3?
IdentityIdentity*batch_normalization_7/FusedBatchNormV3:y:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:?????????R@::::::::::::::::::2n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
??
?3
D__inference_model_layer_call_and_return_conditional_losses_196694366

inputs4
0input_conv_conv2d_conv2d_readvariableop_resource5
1input_conv_conv2d_biasadd_readvariableop_resource:
6input_conv_batch_normalization_readvariableop_resource<
8input_conv_batch_normalization_readvariableop_1_resourceK
Ginput_conv_batch_normalization_fusedbatchnormv3_readvariableop_resourceM
Iinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource6
2input_conv_conv2d_1_conv2d_readvariableop_resource7
3input_conv_conv2d_1_biasadd_readvariableop_resource<
8input_conv_batch_normalization_1_readvariableop_resource>
:input_conv_batch_normalization_1_readvariableop_1_resourceM
Iinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceO
Kinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7
3res_block_0_conv2d_2_conv2d_readvariableop_resource8
4res_block_0_conv2d_2_biasadd_readvariableop_resource=
9res_block_0_batch_normalization_2_readvariableop_resource?
;res_block_0_batch_normalization_2_readvariableop_1_resourceN
Jres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceP
Lres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7
3res_block_0_conv2d_3_conv2d_readvariableop_resource8
4res_block_0_conv2d_3_biasadd_readvariableop_resource=
9res_block_0_batch_normalization_3_readvariableop_resource?
;res_block_0_batch_normalization_3_readvariableop_1_resourceN
Jres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3res_block_0_conv2d_4_conv2d_readvariableop_resource8
4res_block_0_conv2d_4_biasadd_readvariableop_resource=
9res_block_0_batch_normalization_4_readvariableop_resource?
;res_block_0_batch_normalization_4_readvariableop_1_resourceN
Jres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceP
Lres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3res_block_1_conv2d_5_conv2d_readvariableop_resource8
4res_block_1_conv2d_5_biasadd_readvariableop_resource=
9res_block_1_batch_normalization_5_readvariableop_resource?
;res_block_1_batch_normalization_5_readvariableop_1_resourceN
Jres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7
3res_block_1_conv2d_6_conv2d_readvariableop_resource8
4res_block_1_conv2d_6_biasadd_readvariableop_resource=
9res_block_1_batch_normalization_6_readvariableop_resource?
;res_block_1_batch_normalization_6_readvariableop_1_resourceN
Jres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceP
Lres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7
3res_block_1_conv2d_7_conv2d_readvariableop_resource8
4res_block_1_conv2d_7_biasadd_readvariableop_resource=
9res_block_1_batch_normalization_7_readvariableop_resource?
;res_block_1_batch_normalization_7_readvariableop_1_resourceN
Jres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceP
Lres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource3
/feature_conv2d_8_conv2d_readvariableop_resource4
0feature_conv2d_8_biasadd_readvariableop_resource9
5feature_batch_normalization_8_readvariableop_resource;
7feature_batch_normalization_8_readvariableop_1_resourceJ
Ffeature_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceL
Hfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identity??=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp??feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?,feature/batch_normalization_8/ReadVariableOp?.feature/batch_normalization_8/ReadVariableOp_1?'feature/conv2d_8/BiasAdd/ReadVariableOp?&feature/conv2d_8/Conv2D/ReadVariableOp?>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp?@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-input_conv/batch_normalization/ReadVariableOp?/input_conv/batch_normalization/ReadVariableOp_1?@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?/input_conv/batch_normalization_1/ReadVariableOp?1input_conv/batch_normalization_1/ReadVariableOp_1?(input_conv/conv2d/BiasAdd/ReadVariableOp?'input_conv/conv2d/Conv2D/ReadVariableOp?*input_conv/conv2d_1/BiasAdd/ReadVariableOp?)input_conv/conv2d_1/Conv2D/ReadVariableOp?Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?0res_block_0/batch_normalization_2/ReadVariableOp?2res_block_0/batch_normalization_2/ReadVariableOp_1?Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?0res_block_0/batch_normalization_3/ReadVariableOp?2res_block_0/batch_normalization_3/ReadVariableOp_1?Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?0res_block_0/batch_normalization_4/ReadVariableOp?2res_block_0/batch_normalization_4/ReadVariableOp_1?+res_block_0/conv2d_2/BiasAdd/ReadVariableOp?*res_block_0/conv2d_2/Conv2D/ReadVariableOp?+res_block_0/conv2d_3/BiasAdd/ReadVariableOp?*res_block_0/conv2d_3/Conv2D/ReadVariableOp?+res_block_0/conv2d_4/BiasAdd/ReadVariableOp?*res_block_0/conv2d_4/Conv2D/ReadVariableOp?Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?0res_block_1/batch_normalization_5/ReadVariableOp?2res_block_1/batch_normalization_5/ReadVariableOp_1?Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?0res_block_1/batch_normalization_6/ReadVariableOp?2res_block_1/batch_normalization_6/ReadVariableOp_1?Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?0res_block_1/batch_normalization_7/ReadVariableOp?2res_block_1/batch_normalization_7/ReadVariableOp_1?+res_block_1/conv2d_5/BiasAdd/ReadVariableOp?*res_block_1/conv2d_5/Conv2D/ReadVariableOp?+res_block_1/conv2d_6/BiasAdd/ReadVariableOp?*res_block_1/conv2d_6/Conv2D/ReadVariableOp?+res_block_1/conv2d_7/BiasAdd/ReadVariableOp?*res_block_1/conv2d_7/Conv2D/ReadVariableOp?
'input_conv/conv2d/Conv2D/ReadVariableOpReadVariableOp0input_conv_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02)
'input_conv/conv2d/Conv2D/ReadVariableOp?
input_conv/conv2d/Conv2DConv2Dinputs/input_conv/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
input_conv/conv2d/Conv2D?
(input_conv/conv2d/BiasAdd/ReadVariableOpReadVariableOp1input_conv_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(input_conv/conv2d/BiasAdd/ReadVariableOp?
input_conv/conv2d/BiasAddBiasAdd!input_conv/conv2d/Conv2D:output:00input_conv/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
input_conv/conv2d/BiasAdd?
-input_conv/batch_normalization/ReadVariableOpReadVariableOp6input_conv_batch_normalization_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-input_conv/batch_normalization/ReadVariableOp?
/input_conv/batch_normalization/ReadVariableOp_1ReadVariableOp8input_conv_batch_normalization_readvariableop_1_resource*
_output_shapes	
:?*
dtype021
/input_conv/batch_normalization/ReadVariableOp_1?
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02B
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/input_conv/batch_normalization/FusedBatchNormV3FusedBatchNormV3"input_conv/conv2d/BiasAdd:output:05input_conv/batch_normalization/ReadVariableOp:value:07input_conv/batch_normalization/ReadVariableOp_1:value:0Finput_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hinput_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:???????????:?:?:?:?:*
epsilon%o?:*
is_training( 21
/input_conv/batch_normalization/FusedBatchNormV3?
input_conv/activation/ReluRelu3input_conv/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
input_conv/activation/Relu?
)input_conv/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2input_conv_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02+
)input_conv/conv2d_1/Conv2D/ReadVariableOp?
input_conv/conv2d_1/Conv2DConv2D(input_conv/activation/Relu:activations:01input_conv/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingVALID*
strides
2
input_conv/conv2d_1/Conv2D?
*input_conv/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3input_conv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*input_conv/conv2d_1/BiasAdd/ReadVariableOp?
input_conv/conv2d_1/BiasAddBiasAdd#input_conv/conv2d_1/Conv2D:output:02input_conv/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
input_conv/conv2d_1/BiasAdd?
/input_conv/batch_normalization_1/ReadVariableOpReadVariableOp8input_conv_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype021
/input_conv/batch_normalization_1/ReadVariableOp?
1input_conv/batch_normalization_1/ReadVariableOp_1ReadVariableOp:input_conv_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1input_conv/batch_normalization_1/ReadVariableOp_1?
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02B
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
1input_conv/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$input_conv/conv2d_1/BiasAdd:output:07input_conv/batch_normalization_1/ReadVariableOp:value:09input_conv/batch_normalization_1/ReadVariableOp_1:value:0Hinput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jinput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 23
1input_conv/batch_normalization_1/FusedBatchNormV3?
input_conv/activation_1/ReluRelu5input_conv/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
input_conv/activation_1/Relu?
*res_block_0/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_2/Conv2D/ReadVariableOp?
res_block_0/conv2d_2/Conv2DConv2D*input_conv/activation_1/Relu:activations:02res_block_0/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_0/conv2d_2/Conv2D?
+res_block_0/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_2/BiasAdd/ReadVariableOp?
res_block_0/conv2d_2/BiasAddBiasAdd$res_block_0/conv2d_2/Conv2D:output:03res_block_0/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/conv2d_2/BiasAdd?
0res_block_0/batch_normalization_2/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_2/ReadVariableOp?
2res_block_0/batch_normalization_2/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_2/ReadVariableOp_1?
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?
2res_block_0/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_2/BiasAdd:output:08res_block_0/batch_normalization_2/ReadVariableOp:value:0:res_block_0/batch_normalization_2/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_0/batch_normalization_2/FusedBatchNormV3?
res_block_0/activation_2/ReluRelu6res_block_0/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/activation_2/Relu?
*res_block_0/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_3/Conv2D/ReadVariableOp?
res_block_0/conv2d_3/Conv2DConv2D+res_block_0/activation_2/Relu:activations:02res_block_0/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_0/conv2d_3/Conv2D?
+res_block_0/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_3/BiasAdd/ReadVariableOp?
res_block_0/conv2d_3/BiasAddBiasAdd$res_block_0/conv2d_3/Conv2D:output:03res_block_0/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/conv2d_3/BiasAdd?
0res_block_0/batch_normalization_3/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_3/ReadVariableOp?
2res_block_0/batch_normalization_3/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_3/ReadVariableOp_1?
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?
2res_block_0/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_3/BiasAdd:output:08res_block_0/batch_normalization_3/ReadVariableOp:value:0:res_block_0/batch_normalization_3/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_0/batch_normalization_3/FusedBatchNormV3?
res_block_0/activation_3/ReluRelu6res_block_0/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/activation_3/Relu?
*res_block_0/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_4/Conv2D/ReadVariableOp?
res_block_0/conv2d_4/Conv2DConv2D+res_block_0/activation_3/Relu:activations:02res_block_0/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_0/conv2d_4/Conv2D?
+res_block_0/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_4/BiasAdd/ReadVariableOp?
res_block_0/conv2d_4/BiasAddBiasAdd$res_block_0/conv2d_4/Conv2D:output:03res_block_0/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_0/conv2d_4/BiasAdd?
0res_block_0/batch_normalization_4/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_4/ReadVariableOp?
2res_block_0/batch_normalization_4/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_4/ReadVariableOp_1?
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?
2res_block_0/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_4/BiasAdd:output:08res_block_0/batch_normalization_4/ReadVariableOp:value:0:res_block_0/batch_normalization_4/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_0/batch_normalization_4/FusedBatchNormV3?
tf.__operators__.add/AddV2AddV2*input_conv/activation_1/Relu:activations:06res_block_0/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add/AddV2|
relu_0/ReluRelutf.__operators__.add/AddV2:z:0*
T0*/
_output_shapes
:?????????R@2
relu_0/Relu?
*res_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_5/Conv2D/ReadVariableOp?
res_block_1/conv2d_5/Conv2DConv2Drelu_0/Relu:activations:02res_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_1/conv2d_5/Conv2D?
+res_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_5/BiasAdd/ReadVariableOp?
res_block_1/conv2d_5/BiasAddBiasAdd$res_block_1/conv2d_5/Conv2D:output:03res_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/conv2d_5/BiasAdd?
0res_block_1/batch_normalization_5/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_5/ReadVariableOp?
2res_block_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_5/ReadVariableOp_1?
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?
2res_block_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_5/BiasAdd:output:08res_block_1/batch_normalization_5/ReadVariableOp:value:0:res_block_1/batch_normalization_5/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_1/batch_normalization_5/FusedBatchNormV3?
res_block_1/activation_4/ReluRelu6res_block_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/activation_4/Relu?
*res_block_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_6/Conv2D/ReadVariableOp?
res_block_1/conv2d_6/Conv2DConv2D+res_block_1/activation_4/Relu:activations:02res_block_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_1/conv2d_6/Conv2D?
+res_block_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_6/BiasAdd/ReadVariableOp?
res_block_1/conv2d_6/BiasAddBiasAdd$res_block_1/conv2d_6/Conv2D:output:03res_block_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/conv2d_6/BiasAdd?
0res_block_1/batch_normalization_6/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_6/ReadVariableOp?
2res_block_1/batch_normalization_6/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_6/ReadVariableOp_1?
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
2res_block_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_6/BiasAdd:output:08res_block_1/batch_normalization_6/ReadVariableOp:value:0:res_block_1/batch_normalization_6/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_1/batch_normalization_6/FusedBatchNormV3?
res_block_1/activation_5/ReluRelu6res_block_1/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/activation_5/Relu?
*res_block_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_7/Conv2D/ReadVariableOp?
res_block_1/conv2d_7/Conv2DConv2D+res_block_1/activation_5/Relu:activations:02res_block_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@*
paddingSAME*
strides
2
res_block_1/conv2d_7/Conv2D?
+res_block_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_7/BiasAdd/ReadVariableOp?
res_block_1/conv2d_7/BiasAddBiasAdd$res_block_1/conv2d_7/Conv2D:output:03res_block_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R@2
res_block_1/conv2d_7/BiasAdd?
0res_block_1/batch_normalization_7/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_7/ReadVariableOp?
2res_block_1/batch_normalization_7/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_7/ReadVariableOp_1?
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
2res_block_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_7/BiasAdd:output:08res_block_1/batch_normalization_7/ReadVariableOp:value:0:res_block_1/batch_normalization_7/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R@:@:@:@:@:*
epsilon%o?:*
is_training( 24
2res_block_1/batch_normalization_7/FusedBatchNormV3?
tf.__operators__.add_1/AddV2AddV2relu_0/Relu:activations:06res_block_1/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R@2
tf.__operators__.add_1/AddV2~
relu_1/ReluRelu tf.__operators__.add_1/AddV2:z:0*
T0*/
_output_shapes
:?????????R@2
relu_1/Relu?
&feature/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/feature_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&feature/conv2d_8/Conv2D/ReadVariableOp?
feature/conv2d_8/Conv2DConv2Drelu_1/Relu:activations:0.feature/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R*
paddingVALID*
strides
2
feature/conv2d_8/Conv2D?
'feature/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0feature_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'feature/conv2d_8/BiasAdd/ReadVariableOp?
feature/conv2d_8/BiasAddBiasAdd feature/conv2d_8/Conv2D:output:0/feature/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????R2
feature/conv2d_8/BiasAdd?
,feature/batch_normalization_8/ReadVariableOpReadVariableOp5feature_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02.
,feature/batch_normalization_8/ReadVariableOp?
.feature/batch_normalization_8/ReadVariableOp_1ReadVariableOp7feature_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype020
.feature/batch_normalization_8/ReadVariableOp_1?
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
.feature/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3!feature/conv2d_8/BiasAdd:output:04feature/batch_normalization_8/ReadVariableOp:value:06feature/batch_normalization_8/ReadVariableOp_1:value:0Efeature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gfeature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????R:::::*
epsilon%o?:*
is_training( 20
.feature/batch_normalization_8/FusedBatchNormV3?
feature/activation_6/ReluRelu2feature/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????R2
feature/activation_6/Relu?
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/perm?
 tf.compat.v1.transpose/transpose	Transpose'feature/activation_6/Relu:activations:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:?????????R2"
 tf.compat.v1.transpose/transpose?
IdentityIdentity$tf.compat.v1.transpose/transpose:y:0>^feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp@^feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1-^feature/batch_normalization_8/ReadVariableOp/^feature/batch_normalization_8/ReadVariableOp_1(^feature/conv2d_8/BiasAdd/ReadVariableOp'^feature/conv2d_8/Conv2D/ReadVariableOp?^input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpA^input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^input_conv/batch_normalization/ReadVariableOp0^input_conv/batch_normalization/ReadVariableOp_1A^input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^input_conv/batch_normalization_1/ReadVariableOp2^input_conv/batch_normalization_1/ReadVariableOp_1)^input_conv/conv2d/BiasAdd/ReadVariableOp(^input_conv/conv2d/Conv2D/ReadVariableOp+^input_conv/conv2d_1/BiasAdd/ReadVariableOp*^input_conv/conv2d_1/Conv2D/ReadVariableOpB^res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_2/ReadVariableOp3^res_block_0/batch_normalization_2/ReadVariableOp_1B^res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_3/ReadVariableOp3^res_block_0/batch_normalization_3/ReadVariableOp_1B^res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_4/ReadVariableOp3^res_block_0/batch_normalization_4/ReadVariableOp_1,^res_block_0/conv2d_2/BiasAdd/ReadVariableOp+^res_block_0/conv2d_2/Conv2D/ReadVariableOp,^res_block_0/conv2d_3/BiasAdd/ReadVariableOp+^res_block_0/conv2d_3/Conv2D/ReadVariableOp,^res_block_0/conv2d_4/BiasAdd/ReadVariableOp+^res_block_0/conv2d_4/Conv2D/ReadVariableOpB^res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_5/ReadVariableOp3^res_block_1/batch_normalization_5/ReadVariableOp_1B^res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_6/ReadVariableOp3^res_block_1/batch_normalization_6/ReadVariableOp_1B^res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_7/ReadVariableOp3^res_block_1/batch_normalization_7/ReadVariableOp_1,^res_block_1/conv2d_5/BiasAdd/ReadVariableOp+^res_block_1/conv2d_5/Conv2D/ReadVariableOp,^res_block_1/conv2d_6/BiasAdd/ReadVariableOp+^res_block_1/conv2d_6/Conv2D/ReadVariableOp,^res_block_1/conv2d_7/BiasAdd/ReadVariableOp+^res_block_1/conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::::::::::::::::::2~
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12\
,feature/batch_normalization_8/ReadVariableOp,feature/batch_normalization_8/ReadVariableOp2`
.feature/batch_normalization_8/ReadVariableOp_1.feature/batch_normalization_8/ReadVariableOp_12R
'feature/conv2d_8/BiasAdd/ReadVariableOp'feature/conv2d_8/BiasAdd/ReadVariableOp2P
&feature/conv2d_8/Conv2D/ReadVariableOp&feature/conv2d_8/Conv2D/ReadVariableOp2?
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-input_conv/batch_normalization/ReadVariableOp-input_conv/batch_normalization/ReadVariableOp2b
/input_conv/batch_normalization/ReadVariableOp_1/input_conv/batch_normalization/ReadVariableOp_12?
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12b
/input_conv/batch_normalization_1/ReadVariableOp/input_conv/batch_normalization_1/ReadVariableOp2f
1input_conv/batch_normalization_1/ReadVariableOp_11input_conv/batch_normalization_1/ReadVariableOp_12T
(input_conv/conv2d/BiasAdd/ReadVariableOp(input_conv/conv2d/BiasAdd/ReadVariableOp2R
'input_conv/conv2d/Conv2D/ReadVariableOp'input_conv/conv2d/Conv2D/ReadVariableOp2X
*input_conv/conv2d_1/BiasAdd/ReadVariableOp*input_conv/conv2d_1/BiasAdd/ReadVariableOp2V
)input_conv/conv2d_1/Conv2D/ReadVariableOp)input_conv/conv2d_1/Conv2D/ReadVariableOp2?
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2?
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_2/ReadVariableOp0res_block_0/batch_normalization_2/ReadVariableOp2h
2res_block_0/batch_normalization_2/ReadVariableOp_12res_block_0/batch_normalization_2/ReadVariableOp_12?
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_3/ReadVariableOp0res_block_0/batch_normalization_3/ReadVariableOp2h
2res_block_0/batch_normalization_3/ReadVariableOp_12res_block_0/batch_normalization_3/ReadVariableOp_12?
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_4/ReadVariableOp0res_block_0/batch_normalization_4/ReadVariableOp2h
2res_block_0/batch_normalization_4/ReadVariableOp_12res_block_0/batch_normalization_4/ReadVariableOp_12Z
+res_block_0/conv2d_2/BiasAdd/ReadVariableOp+res_block_0/conv2d_2/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_2/Conv2D/ReadVariableOp*res_block_0/conv2d_2/Conv2D/ReadVariableOp2Z
+res_block_0/conv2d_3/BiasAdd/ReadVariableOp+res_block_0/conv2d_3/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_3/Conv2D/ReadVariableOp*res_block_0/conv2d_3/Conv2D/ReadVariableOp2Z
+res_block_0/conv2d_4/BiasAdd/ReadVariableOp+res_block_0/conv2d_4/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_4/Conv2D/ReadVariableOp*res_block_0/conv2d_4/Conv2D/ReadVariableOp2?
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_5/ReadVariableOp0res_block_1/batch_normalization_5/ReadVariableOp2h
2res_block_1/batch_normalization_5/ReadVariableOp_12res_block_1/batch_normalization_5/ReadVariableOp_12?
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_6/ReadVariableOp0res_block_1/batch_normalization_6/ReadVariableOp2h
2res_block_1/batch_normalization_6/ReadVariableOp_12res_block_1/batch_normalization_6/ReadVariableOp_12?
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_7/ReadVariableOp0res_block_1/batch_normalization_7/ReadVariableOp2h
2res_block_1/batch_normalization_7/ReadVariableOp_12res_block_1/batch_normalization_7/ReadVariableOp_12Z
+res_block_1/conv2d_5/BiasAdd/ReadVariableOp+res_block_1/conv2d_5/BiasAdd/ReadVariableOp2X
*res_block_1/conv2d_5/Conv2D/ReadVariableOp*res_block_1/conv2d_5/Conv2D/ReadVariableOp2Z
+res_block_1/conv2d_6/BiasAdd/ReadVariableOp+res_block_1/conv2d_6/BiasAdd/ReadVariableOp2X
*res_block_1/conv2d_6/Conv2D/ReadVariableOp*res_block_1/conv2d_6/Conv2D/ReadVariableOp2Z
+res_block_1/conv2d_7/BiasAdd/ReadVariableOp+res_block_1/conv2d_7/BiasAdd/ReadVariableOp2X
*res_block_1/conv2d_7/Conv2D/ReadVariableOp*res_block_1/conv2d_7/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196690510

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,????????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196691296

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
,__inference_conv2d_5_layer_call_fn_196696254

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
:?????????R@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_5_layer_call_and_return_conditional_losses_1966922012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
F__inference_feature_layer_call_and_return_conditional_losses_196693028

inputs
conv2d_8_196693012
conv2d_8_196693014#
batch_normalization_8_196693017#
batch_normalization_8_196693019#
batch_normalization_8_196693021#
batch_normalization_8_196693023
identity??-batch_normalization_8/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_196693012conv2d_8_196693014*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_8_layer_call_and_return_conditional_losses_1966928502"
 conv2d_8/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_196693017batch_normalization_8_196693019batch_normalization_8_196693021batch_normalization_8_196693023*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1966929012/
-batch_normalization_8/StatefulPartitionedCall?
activation_6/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????R* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_activation_6_layer_call_and_return_conditional_losses_1966929422
activation_6/PartitionedCall?
IdentityIdentity%activation_6/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????R@::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:W S
/
_output_shapes
:?????????R@
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196691096

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input7
serving_default_input:0??????????J
feature_linear8
StatefulPartitionedCall:0?????????Rtensorflow/serving/predict:??

??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10

signatures
#_self_saveable_object_factories
	optimizer
loss
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"??
_tf_keras_network??{"class_name": "Functional", "name": "model", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "input_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "name": "input_conv", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "res_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "res_block_0", "inbound_nodes": [[["input_conv", 1, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": false, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["input_conv", 1, 0, {"y": ["res_block_0", 1, 0], "name": null}]]}, {"class_name": "Activation", "config": {"name": "relu_0", "trainable": false, "dtype": "float32", "activation": "relu"}, "name": "relu_0", "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "res_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "res_block_1", "inbound_nodes": [[["relu_0", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": false, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["relu_0", 0, 0, {"y": ["res_block_1", 1, 0], "name": null}]]}, {"class_name": "Activation", "config": {"name": "relu_1", "trainable": false, "dtype": "float32", "activation": "relu"}, "name": "relu_1", "inbound_nodes": [[["tf.__operators__.add_1", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "feature", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "name": "feature", "inbound_nodes": [[["relu_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.transpose", "trainable": false, "dtype": "float32", "function": "compat.v1.transpose"}, "name": "tf.compat.v1.transpose", "inbound_nodes": [["feature", 1, 0, {"perm": {"class_name": "__tuple__", "items": [0, 1, 3, 2]}}]]}, {"class_name": "Activation", "config": {"name": "feature_linear", "trainable": false, "dtype": "float32", "activation": "linear"}, "name": "feature_linear", "inbound_nodes": [[["tf.compat.v1.transpose", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["feature_linear", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 508, 8]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "input_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "name": "input_conv", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "res_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "res_block_0", "inbound_nodes": [[["input_conv", 1, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": false, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["input_conv", 1, 0, {"y": ["res_block_0", 1, 0], "name": null}]]}, {"class_name": "Activation", "config": {"name": "relu_0", "trainable": false, "dtype": "float32", "activation": "relu"}, "name": "relu_0", "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "res_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "res_block_1", "inbound_nodes": [[["relu_0", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": false, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["relu_0", 0, 0, {"y": ["res_block_1", 1, 0], "name": null}]]}, {"class_name": "Activation", "config": {"name": "relu_1", "trainable": false, "dtype": "float32", "activation": "relu"}, "name": "relu_1", "inbound_nodes": [[["tf.__operators__.add_1", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "feature", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "name": "feature", "inbound_nodes": [[["relu_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.transpose", "trainable": false, "dtype": "float32", "function": "compat.v1.transpose"}, "name": "tf.compat.v1.transpose", "inbound_nodes": [["feature", 1, 0, {"perm": {"class_name": "__tuple__", "items": [0, 1, 3, 2]}}]]}, {"class_name": "Activation", "config": {"name": "feature_linear", "trainable": false, "dtype": "float32", "activation": "linear"}, "name": "feature_linear", "inbound_nodes": [[["tf.compat.v1.transpose", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["feature_linear", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.001, "decay": 0.0, "rho": 0.9, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
?
#_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
?5
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?2
_tf_keras_sequential?2{"class_name": "Sequential", "name": "input_conv", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "input_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 508, 8]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "input_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}]}}}
?I
 layer_with_weights-0
 layer-0
!layer_with_weights-1
!layer-1
"layer-2
#layer_with_weights-2
#layer-3
$layer_with_weights-3
$layer-4
%layer-5
&layer_with_weights-4
&layer-6
'layer_with_weights-5
'layer-7
#(_self_saveable_object_factories
)regularization_losses
*trainable_variables
+	variables
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?F
_tf_keras_sequential?F{"class_name": "Sequential", "name": "res_block_0", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "res_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "res_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}}}
?
#-_self_saveable_object_factories
.	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add", "trainable": false, "dtype": "float32", "function": "__operators__.add"}}
?
#/_self_saveable_object_factories
0regularization_losses
1trainable_variables
2	variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "relu_0", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu_0", "trainable": false, "dtype": "float32", "activation": "relu"}}
?I
4layer_with_weights-0
4layer-0
5layer_with_weights-1
5layer-1
6layer-2
7layer_with_weights-2
7layer-3
8layer_with_weights-3
8layer-4
9layer-5
:layer_with_weights-4
:layer-6
;layer_with_weights-5
;layer-7
#<_self_saveable_object_factories
=regularization_losses
>trainable_variables
?	variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?F
_tf_keras_sequential?F{"class_name": "Sequential", "name": "res_block_1", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "res_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "res_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}}}
?
#A_self_saveable_object_factories
B	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_1", "trainable": false, "dtype": "float32", "function": "__operators__.add"}}
?
#C_self_saveable_object_factories
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "relu_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu_1", "trainable": false, "dtype": "float32", "activation": "relu"}}
?
Hlayer_with_weights-0
Hlayer-0
Ilayer_with_weights-1
Ilayer-1
Jlayer-2
#K_self_saveable_object_factories
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "feature", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "feature", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "feature", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}]}}}
?
#P_self_saveable_object_factories
Q	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.transpose", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.transpose", "trainable": false, "dtype": "float32", "function": "compat.v1.transpose"}}
?
#R_self_saveable_object_factories
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "feature_linear", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "feature_linear", "trainable": false, "dtype": "float32", "activation": "linear"}}
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
"
	optimizer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11
c12
d13
e14
f15
g16
h17
i18
j19
k20
l21
m22
n23
o24
p25
q26
r27
s28
t29
u30
v31
w32
x33
y34
z35
{36
|37
}38
~39
40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53"
trackable_list_wrapper
?
?layers
regularization_losses
 ?layer_regularization_losses
?layer_metrics
trainable_variables
?non_trainable_variables
	variables
?metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
?


Wkernel
Xbias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 508, 8]}}
?	
	?axis
	Ygamma
Zbeta
[moving_mean
\moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 168, 128]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}
?


]kernel
^bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 168, 128]}}
?	
	?axis
	_gamma
`beta
amoving_mean
bmoving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
v
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11"
trackable_list_wrapper
?
?layers
regularization_losses
 ?layer_regularization_losses
?layer_metrics
trainable_variables
?non_trainable_variables
	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?


ckernel
dbias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?	
	?axis
	egamma
fbeta
gmoving_mean
hmoving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}
?


ikernel
jbias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?	
	?axis
	kgamma
lbeta
mmoving_mean
nmoving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}
?


okernel
pbias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?	
	?axis
	qgamma
rbeta
smoving_mean
tmoving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15
s16
t17"
trackable_list_wrapper
?
?layers
)regularization_losses
 ?layer_regularization_losses
?layer_metrics
*trainable_variables
?non_trainable_variables
+	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
"
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
?layers
0regularization_losses
 ?layer_regularization_losses
?layer_metrics
1trainable_variables
?non_trainable_variables
2	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?


ukernel
vbias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_5", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?	
	?axis
	wgamma
xbeta
ymoving_mean
zmoving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}
?


{kernel
|bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_6", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?	
	?axis
	}gamma
~beta
moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_5", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}
?

?kernel
	?bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_7", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
?11
?12
?13
?14
?15
?16
?17"
trackable_list_wrapper
?
?layers
=regularization_losses
 ?layer_regularization_losses
?layer_metrics
>trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
"
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
?layers
Dregularization_losses
 ?layer_regularization_losses
?layer_metrics
Etrainable_variables
?non_trainable_variables
F	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_8", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?	
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 8]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_6", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
?
?layers
Lregularization_losses
 ?layer_regularization_losses
?layer_metrics
Mtrainable_variables
?non_trainable_variables
N	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
"
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
?layers
Sregularization_losses
 ?layer_regularization_losses
?layer_metrics
Ttrainable_variables
?non_trainable_variables
U	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&?2conv2d/kernel
:?2conv2d/bias
(:&?2batch_normalization/gamma
':%?2batch_normalization/beta
0:.? (2batch_normalization/moving_mean
4:2? (2#batch_normalization/moving_variance
*:(?@2conv2d_1/kernel
:@2conv2d_1/bias
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
):'@@2conv2d_3/kernel
:@2conv2d_3/bias
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
):'@@2conv2d_4/kernel
:@2conv2d_4/bias
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
):'@@2conv2d_5/kernel
:@2conv2d_5/bias
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
):'@@2conv2d_6/kernel
:@2conv2d_6/bias
):'@2batch_normalization_6/gamma
(:&@2batch_normalization_6/beta
1:/@ (2!batch_normalization_6/moving_mean
5:3@ (2%batch_normalization_6/moving_variance
):'@@2conv2d_7/kernel
:@2conv2d_7/bias
):'@2batch_normalization_7/gamma
(:&@2batch_normalization_7/beta
1:/@ (2!batch_normalization_7/moving_mean
5:3@ (2%batch_normalization_7/moving_variance
):'@2conv2d_8/kernel
:2conv2d_8/bias
):'2batch_normalization_8/gamma
(:&2batch_normalization_8/beta
1:/ (2!batch_normalization_8/moving_mean
5:3 (2%batch_normalization_8/moving_variance
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11
c12
d13
e14
f15
g16
h17
i18
j19
k20
l21
m22
n23
o24
p25
q26
r27
s28
t29
u30
v31
w32
x33
y34
z35
{36
|37
}38
~39
40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
Y0
Z1
[2
\3"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
_0
`1
a2
b3"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
v
W0
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
e0
f1
g2
h3"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
k0
l1
m2
n3"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
q0
r1
s2
t3"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
X
 0
!1
"2
#3
$4
%5
&6
'7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
o12
p13
q14
r15
s16
t17"
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
w0
x1
y2
z3"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
=
}0
~1
2
?3"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
X
40
51
62
73
84
95
:6
;7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
?11
?12
?13
?14
?15
?16
?17"
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?layers
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?non_trainable_variables
?	variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5
H0
I1
J2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
P
?0
?1
?2
?3
?4
?5"
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
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
Y0
Z1
[2
\3"
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
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
_0
`1
a2
b3"
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
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
e0
f1
g2
h3"
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
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
k0
l1
m2
n3"
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
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
q0
r1
s2
t3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
w0
x1
y2
z3"
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
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
=
}0
~1
2
?3"
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
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
@
?0
?1
?2
?3"
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
)__inference_model_layer_call_fn_196694789
)__inference_model_layer_call_fn_196694054
)__inference_model_layer_call_fn_196693818
)__inference_model_layer_call_fn_196694676?
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
?2?
D__inference_model_layer_call_and_return_conditional_losses_196693581
D__inference_model_layer_call_and_return_conditional_losses_196693458
D__inference_model_layer_call_and_return_conditional_losses_196694366
D__inference_model_layer_call_and_return_conditional_losses_196694563?
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
$__inference__wrapped_model_196690452?
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
annotations? *-?*
(?%
input??????????
?2?
.__inference_input_conv_layer_call_fn_196694910
.__inference_input_conv_layer_call_fn_196690975
.__inference_input_conv_layer_call_fn_196691038
.__inference_input_conv_layer_call_fn_196694939?
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
?2?
I__inference_input_conv_layer_call_and_return_conditional_losses_196694835
I__inference_input_conv_layer_call_and_return_conditional_losses_196690911
I__inference_input_conv_layer_call_and_return_conditional_losses_196694881
I__inference_input_conv_layer_call_and_return_conditional_losses_196690877?
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
?2?
/__inference_res_block_0_layer_call_fn_196691887
/__inference_res_block_0_layer_call_fn_196695153
/__inference_res_block_0_layer_call_fn_196695112
/__inference_res_block_0_layer_call_fn_196691798?
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
?2?
J__inference_res_block_0_layer_call_and_return_conditional_losses_196695005
J__inference_res_block_0_layer_call_and_return_conditional_losses_196691708
J__inference_res_block_0_layer_call_and_return_conditional_losses_196695071
J__inference_res_block_0_layer_call_and_return_conditional_losses_196691660?
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
*__inference_relu_0_layer_call_fn_196695163?
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
E__inference_relu_0_layer_call_and_return_conditional_losses_196695158?
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
?2?
/__inference_res_block_1_layer_call_fn_196695336
/__inference_res_block_1_layer_call_fn_196695377
/__inference_res_block_1_layer_call_fn_196692736
/__inference_res_block_1_layer_call_fn_196692647?
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
?2?
J__inference_res_block_1_layer_call_and_return_conditional_losses_196695229
J__inference_res_block_1_layer_call_and_return_conditional_losses_196692509
J__inference_res_block_1_layer_call_and_return_conditional_losses_196695295
J__inference_res_block_1_layer_call_and_return_conditional_losses_196692557?
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
*__inference_relu_1_layer_call_fn_196695387?
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
E__inference_relu_1_layer_call_and_return_conditional_losses_196695382?
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
+__inference_feature_layer_call_fn_196695471
+__inference_feature_layer_call_fn_196693043
+__inference_feature_layer_call_fn_196693007
+__inference_feature_layer_call_fn_196695454?
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
?2?
F__inference_feature_layer_call_and_return_conditional_losses_196695437
F__inference_feature_layer_call_and_return_conditional_losses_196695412
F__inference_feature_layer_call_and_return_conditional_losses_196692970
F__inference_feature_layer_call_and_return_conditional_losses_196692951?
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
2__inference_feature_linear_layer_call_fn_196695480?
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
M__inference_feature_linear_layer_call_and_return_conditional_losses_196695475?
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
'__inference_signature_wrapper_196694169input"?
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
 
?2?
*__inference_conv2d_layer_call_fn_196695499?
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
E__inference_conv2d_layer_call_and_return_conditional_losses_196695490?
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
?2?
7__inference_batch_normalization_layer_call_fn_196695561
7__inference_batch_normalization_layer_call_fn_196695623
7__inference_batch_normalization_layer_call_fn_196695548
7__inference_batch_normalization_layer_call_fn_196695610?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196695517
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196695597
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196695579
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196695535?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_activation_layer_call_fn_196695633?
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
I__inference_activation_layer_call_and_return_conditional_losses_196695628?
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
,__inference_conv2d_1_layer_call_fn_196695652?
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
G__inference_conv2d_1_layer_call_and_return_conditional_losses_196695643?
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
?2?
9__inference_batch_normalization_1_layer_call_fn_196695763
9__inference_batch_normalization_1_layer_call_fn_196695701
9__inference_batch_normalization_1_layer_call_fn_196695776
9__inference_batch_normalization_1_layer_call_fn_196695714?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196695750
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196695670
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196695688
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196695732?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_activation_1_layer_call_fn_196695786?
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
K__inference_activation_1_layer_call_and_return_conditional_losses_196695781?
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
,__inference_conv2d_2_layer_call_fn_196695805?
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
G__inference_conv2d_2_layer_call_and_return_conditional_losses_196695796?
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
?2?
9__inference_batch_normalization_2_layer_call_fn_196695916
9__inference_batch_normalization_2_layer_call_fn_196695929
9__inference_batch_normalization_2_layer_call_fn_196695867
9__inference_batch_normalization_2_layer_call_fn_196695854?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196695885
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196695841
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196695823
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196695903?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_activation_2_layer_call_fn_196695939?
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
K__inference_activation_2_layer_call_and_return_conditional_losses_196695934?
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
,__inference_conv2d_3_layer_call_fn_196695958?
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
G__inference_conv2d_3_layer_call_and_return_conditional_losses_196695949?
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
?2?
9__inference_batch_normalization_3_layer_call_fn_196696020
9__inference_batch_normalization_3_layer_call_fn_196696069
9__inference_batch_normalization_3_layer_call_fn_196696082
9__inference_batch_normalization_3_layer_call_fn_196696007?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196696056
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196695994
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196696038
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196695976?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_activation_3_layer_call_fn_196696092?
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
K__inference_activation_3_layer_call_and_return_conditional_losses_196696087?
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
,__inference_conv2d_4_layer_call_fn_196696111?
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
G__inference_conv2d_4_layer_call_and_return_conditional_losses_196696102?
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
?2?
9__inference_batch_normalization_4_layer_call_fn_196696222
9__inference_batch_normalization_4_layer_call_fn_196696160
9__inference_batch_normalization_4_layer_call_fn_196696235
9__inference_batch_normalization_4_layer_call_fn_196696173?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196696147
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196696129
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196696209
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196696191?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_conv2d_5_layer_call_fn_196696254?
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
G__inference_conv2d_5_layer_call_and_return_conditional_losses_196696245?
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
?2?
9__inference_batch_normalization_5_layer_call_fn_196696303
9__inference_batch_normalization_5_layer_call_fn_196696365
9__inference_batch_normalization_5_layer_call_fn_196696316
9__inference_batch_normalization_5_layer_call_fn_196696378?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196696290
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196696334
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196696272
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196696352?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_activation_4_layer_call_fn_196696388?
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
K__inference_activation_4_layer_call_and_return_conditional_losses_196696383?
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
,__inference_conv2d_6_layer_call_fn_196696407?
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
G__inference_conv2d_6_layer_call_and_return_conditional_losses_196696398?
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
?2?
9__inference_batch_normalization_6_layer_call_fn_196696531
9__inference_batch_normalization_6_layer_call_fn_196696518
9__inference_batch_normalization_6_layer_call_fn_196696456
9__inference_batch_normalization_6_layer_call_fn_196696469?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196696505
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196696425
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196696487
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196696443?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_activation_5_layer_call_fn_196696541?
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
K__inference_activation_5_layer_call_and_return_conditional_losses_196696536?
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
,__inference_conv2d_7_layer_call_fn_196696560?
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
G__inference_conv2d_7_layer_call_and_return_conditional_losses_196696551?
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
?2?
9__inference_batch_normalization_7_layer_call_fn_196696609
9__inference_batch_normalization_7_layer_call_fn_196696684
9__inference_batch_normalization_7_layer_call_fn_196696622
9__inference_batch_normalization_7_layer_call_fn_196696671?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196696596
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196696658
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196696578
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196696640?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_conv2d_8_layer_call_fn_196696703?
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
G__inference_conv2d_8_layer_call_and_return_conditional_losses_196696694?
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
?2?
9__inference_batch_normalization_8_layer_call_fn_196696752
9__inference_batch_normalization_8_layer_call_fn_196696827
9__inference_batch_normalization_8_layer_call_fn_196696814
9__inference_batch_normalization_8_layer_call_fn_196696765?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196696801
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196696739
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196696783
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196696721?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_activation_6_layer_call_fn_196696837?
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
K__inference_activation_6_layer_call_and_return_conditional_losses_196696832?
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
 ?
$__inference__wrapped_model_196690452?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????7?4
-?*
(?%
input??????????
? "G?D
B
feature_linear0?-
feature_linear?????????R?
K__inference_activation_1_layer_call_and_return_conditional_losses_196695781h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
0__inference_activation_1_layer_call_fn_196695786[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
K__inference_activation_2_layer_call_and_return_conditional_losses_196695934h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
0__inference_activation_2_layer_call_fn_196695939[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
K__inference_activation_3_layer_call_and_return_conditional_losses_196696087h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
0__inference_activation_3_layer_call_fn_196696092[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
K__inference_activation_4_layer_call_and_return_conditional_losses_196696383h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
0__inference_activation_4_layer_call_fn_196696388[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
K__inference_activation_5_layer_call_and_return_conditional_losses_196696536h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
0__inference_activation_5_layer_call_fn_196696541[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
K__inference_activation_6_layer_call_and_return_conditional_losses_196696832h7?4
-?*
(?%
inputs?????????R
? "-?*
#? 
0?????????R
? ?
0__inference_activation_6_layer_call_fn_196696837[7?4
-?*
(?%
inputs?????????R
? " ??????????R?
I__inference_activation_layer_call_and_return_conditional_losses_196695628l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
.__inference_activation_layer_call_fn_196695633_9?6
/?,
*?'
inputs???????????
? ""?????????????
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196695670?_`abM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196695688?_`abM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196695732r_`ab;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_196695750r_`ab;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
9__inference_batch_normalization_1_layer_call_fn_196695701?_`abM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
9__inference_batch_normalization_1_layer_call_fn_196695714?_`abM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
9__inference_batch_normalization_1_layer_call_fn_196695763e_`ab;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
9__inference_batch_normalization_1_layer_call_fn_196695776e_`ab;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196695823?efghM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196695841?efghM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196695885refgh;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_196695903refgh;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
9__inference_batch_normalization_2_layer_call_fn_196695854?efghM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
9__inference_batch_normalization_2_layer_call_fn_196695867?efghM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
9__inference_batch_normalization_2_layer_call_fn_196695916eefgh;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
9__inference_batch_normalization_2_layer_call_fn_196695929eefgh;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196695976rklmn;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196695994rklmn;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196696038?klmnM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_196696056?klmnM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
9__inference_batch_normalization_3_layer_call_fn_196696007eklmn;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
9__inference_batch_normalization_3_layer_call_fn_196696020eklmn;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
9__inference_batch_normalization_3_layer_call_fn_196696069?klmnM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
9__inference_batch_normalization_3_layer_call_fn_196696082?klmnM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196696129?qrstM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196696147?qrstM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196696191rqrst;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_196696209rqrst;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
9__inference_batch_normalization_4_layer_call_fn_196696160?qrstM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
9__inference_batch_normalization_4_layer_call_fn_196696173?qrstM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
9__inference_batch_normalization_4_layer_call_fn_196696222eqrst;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
9__inference_batch_normalization_4_layer_call_fn_196696235eqrst;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196696272?wxyzM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196696290?wxyzM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196696334rwxyz;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_196696352rwxyz;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
9__inference_batch_normalization_5_layer_call_fn_196696303?wxyzM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
9__inference_batch_normalization_5_layer_call_fn_196696316?wxyzM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
9__inference_batch_normalization_5_layer_call_fn_196696365ewxyz;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
9__inference_batch_normalization_5_layer_call_fn_196696378ewxyz;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196696425s}~?;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196696443s}~?;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196696487?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_196696505?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
9__inference_batch_normalization_6_layer_call_fn_196696456f}~?;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
9__inference_batch_normalization_6_layer_call_fn_196696469f}~?;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
9__inference_batch_normalization_6_layer_call_fn_196696518?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
9__inference_batch_normalization_6_layer_call_fn_196696531?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196696578?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196696596?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196696640v????;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_196696658v????;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
9__inference_batch_normalization_7_layer_call_fn_196696609?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
9__inference_batch_normalization_7_layer_call_fn_196696622?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
9__inference_batch_normalization_7_layer_call_fn_196696671i????;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
9__inference_batch_normalization_7_layer_call_fn_196696684i????;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196696721v????;?8
1?.
(?%
inputs?????????R
p
? "-?*
#? 
0?????????R
? ?
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196696739v????;?8
1?.
(?%
inputs?????????R
p 
? "-?*
#? 
0?????????R
? ?
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196696783?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_196696801?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
9__inference_batch_normalization_8_layer_call_fn_196696752i????;?8
1?.
(?%
inputs?????????R
p
? " ??????????R?
9__inference_batch_normalization_8_layer_call_fn_196696765i????;?8
1?.
(?%
inputs?????????R
p 
? " ??????????R?
9__inference_batch_normalization_8_layer_call_fn_196696814?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
9__inference_batch_normalization_8_layer_call_fn_196696827?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196695517vYZ[\=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196695535vYZ[\=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196695579?YZ[\N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_layer_call_and_return_conditional_losses_196695597?YZ[\N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
7__inference_batch_normalization_layer_call_fn_196695548iYZ[\=?:
3?0
*?'
inputs???????????
p
? ""?????????????
7__inference_batch_normalization_layer_call_fn_196695561iYZ[\=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
7__inference_batch_normalization_layer_call_fn_196695610?YZ[\N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
7__inference_batch_normalization_layer_call_fn_196695623?YZ[\N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
G__inference_conv2d_1_layer_call_and_return_conditional_losses_196695643n]^9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????R@
? ?
,__inference_conv2d_1_layer_call_fn_196695652a]^9?6
/?,
*?'
inputs???????????
? " ??????????R@?
G__inference_conv2d_2_layer_call_and_return_conditional_losses_196695796lcd7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
,__inference_conv2d_2_layer_call_fn_196695805_cd7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
G__inference_conv2d_3_layer_call_and_return_conditional_losses_196695949lij7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
,__inference_conv2d_3_layer_call_fn_196695958_ij7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
G__inference_conv2d_4_layer_call_and_return_conditional_losses_196696102lop7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
,__inference_conv2d_4_layer_call_fn_196696111_op7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
G__inference_conv2d_5_layer_call_and_return_conditional_losses_196696245luv7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
,__inference_conv2d_5_layer_call_fn_196696254_uv7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
G__inference_conv2d_6_layer_call_and_return_conditional_losses_196696398l{|7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
,__inference_conv2d_6_layer_call_fn_196696407_{|7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
G__inference_conv2d_7_layer_call_and_return_conditional_losses_196696551n??7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
,__inference_conv2d_7_layer_call_fn_196696560a??7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
G__inference_conv2d_8_layer_call_and_return_conditional_losses_196696694n??7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R
? ?
,__inference_conv2d_8_layer_call_fn_196696703a??7?4
-?*
(?%
inputs?????????R@
? " ??????????R?
E__inference_conv2d_layer_call_and_return_conditional_losses_196695490oWX8?5
.?+
)?&
inputs??????????
? "/?,
%?"
0???????????
? ?
*__inference_conv2d_layer_call_fn_196695499bWX8?5
.?+
)?&
inputs??????????
? ""?????????????
F__inference_feature_layer_call_and_return_conditional_losses_196692951???????G?D
=?:
0?-
conv2d_8_input?????????R@
p

 
? "-?*
#? 
0?????????R
? ?
F__inference_feature_layer_call_and_return_conditional_losses_196692970???????G?D
=?:
0?-
conv2d_8_input?????????R@
p 

 
? "-?*
#? 
0?????????R
? ?
F__inference_feature_layer_call_and_return_conditional_losses_196695412~????????<
5?2
(?%
inputs?????????R@
p

 
? "-?*
#? 
0?????????R
? ?
F__inference_feature_layer_call_and_return_conditional_losses_196695437~????????<
5?2
(?%
inputs?????????R@
p 

 
? "-?*
#? 
0?????????R
? ?
+__inference_feature_layer_call_fn_196693007y??????G?D
=?:
0?-
conv2d_8_input?????????R@
p

 
? " ??????????R?
+__inference_feature_layer_call_fn_196693043y??????G?D
=?:
0?-
conv2d_8_input?????????R@
p 

 
? " ??????????R?
+__inference_feature_layer_call_fn_196695454q????????<
5?2
(?%
inputs?????????R@
p

 
? " ??????????R?
+__inference_feature_layer_call_fn_196695471q????????<
5?2
(?%
inputs?????????R@
p 

 
? " ??????????R?
M__inference_feature_linear_layer_call_and_return_conditional_losses_196695475h7?4
-?*
(?%
inputs?????????R
? "-?*
#? 
0?????????R
? ?
2__inference_feature_linear_layer_call_fn_196695480[7?4
-?*
(?%
inputs?????????R
? " ??????????R?
I__inference_input_conv_layer_call_and_return_conditional_losses_196690877?WXYZ[\]^_`abF?C
<?9
/?,
conv2d_input??????????
p

 
? "-?*
#? 
0?????????R@
? ?
I__inference_input_conv_layer_call_and_return_conditional_losses_196690911?WXYZ[\]^_`abF?C
<?9
/?,
conv2d_input??????????
p 

 
? "-?*
#? 
0?????????R@
? ?
I__inference_input_conv_layer_call_and_return_conditional_losses_196694835WXYZ[\]^_`ab@?=
6?3
)?&
inputs??????????
p

 
? "-?*
#? 
0?????????R@
? ?
I__inference_input_conv_layer_call_and_return_conditional_losses_196694881WXYZ[\]^_`ab@?=
6?3
)?&
inputs??????????
p 

 
? "-?*
#? 
0?????????R@
? ?
.__inference_input_conv_layer_call_fn_196690975xWXYZ[\]^_`abF?C
<?9
/?,
conv2d_input??????????
p

 
? " ??????????R@?
.__inference_input_conv_layer_call_fn_196691038xWXYZ[\]^_`abF?C
<?9
/?,
conv2d_input??????????
p 

 
? " ??????????R@?
.__inference_input_conv_layer_call_fn_196694910rWXYZ[\]^_`ab@?=
6?3
)?&
inputs??????????
p

 
? " ??????????R@?
.__inference_input_conv_layer_call_fn_196694939rWXYZ[\]^_`ab@?=
6?3
)?&
inputs??????????
p 

 
? " ??????????R@?
D__inference_model_layer_call_and_return_conditional_losses_196693458?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????????<
5?2
(?%
input??????????
p

 
? "-?*
#? 
0?????????R
? ?
D__inference_model_layer_call_and_return_conditional_losses_196693581?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????????<
5?2
(?%
input??????????
p 

 
? "-?*
#? 
0?????????R
? ?
D__inference_model_layer_call_and_return_conditional_losses_196694366?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
6?3
)?&
inputs??????????
p

 
? "-?*
#? 
0?????????R
? ?
D__inference_model_layer_call_and_return_conditional_losses_196694563?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
6?3
)?&
inputs??????????
p 

 
? "-?*
#? 
0?????????R
? ?
)__inference_model_layer_call_fn_196693818?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????????<
5?2
(?%
input??????????
p

 
? " ??????????R?
)__inference_model_layer_call_fn_196694054?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????????<
5?2
(?%
input??????????
p 

 
? " ??????????R?
)__inference_model_layer_call_fn_196694676?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
6?3
)?&
inputs??????????
p

 
? " ??????????R?
)__inference_model_layer_call_fn_196694789?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
6?3
)?&
inputs??????????
p 

 
? " ??????????R?
E__inference_relu_0_layer_call_and_return_conditional_losses_196695158h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
*__inference_relu_0_layer_call_fn_196695163[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
E__inference_relu_1_layer_call_and_return_conditional_losses_196695382h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
*__inference_relu_1_layer_call_fn_196695387[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
J__inference_res_block_0_layer_call_and_return_conditional_losses_196691660?cdefghijklmnopqrstG?D
=?:
0?-
conv2d_2_input?????????R@
p

 
? "-?*
#? 
0?????????R@
? ?
J__inference_res_block_0_layer_call_and_return_conditional_losses_196691708?cdefghijklmnopqrstG?D
=?:
0?-
conv2d_2_input?????????R@
p 

 
? "-?*
#? 
0?????????R@
? ?
J__inference_res_block_0_layer_call_and_return_conditional_losses_196695005?cdefghijklmnopqrst??<
5?2
(?%
inputs?????????R@
p

 
? "-?*
#? 
0?????????R@
? ?
J__inference_res_block_0_layer_call_and_return_conditional_losses_196695071?cdefghijklmnopqrst??<
5?2
(?%
inputs?????????R@
p 

 
? "-?*
#? 
0?????????R@
? ?
/__inference_res_block_0_layer_call_fn_196691798cdefghijklmnopqrstG?D
=?:
0?-
conv2d_2_input?????????R@
p

 
? " ??????????R@?
/__inference_res_block_0_layer_call_fn_196691887cdefghijklmnopqrstG?D
=?:
0?-
conv2d_2_input?????????R@
p 

 
? " ??????????R@?
/__inference_res_block_0_layer_call_fn_196695112wcdefghijklmnopqrst??<
5?2
(?%
inputs?????????R@
p

 
? " ??????????R@?
/__inference_res_block_0_layer_call_fn_196695153wcdefghijklmnopqrst??<
5?2
(?%
inputs?????????R@
p 

 
? " ??????????R@?
J__inference_res_block_1_layer_call_and_return_conditional_losses_196692509?uvwxyz{|}~???????G?D
=?:
0?-
conv2d_5_input?????????R@
p

 
? "-?*
#? 
0?????????R@
? ?
J__inference_res_block_1_layer_call_and_return_conditional_losses_196692557?uvwxyz{|}~???????G?D
=?:
0?-
conv2d_5_input?????????R@
p 

 
? "-?*
#? 
0?????????R@
? ?
J__inference_res_block_1_layer_call_and_return_conditional_losses_196695229?uvwxyz{|}~?????????<
5?2
(?%
inputs?????????R@
p

 
? "-?*
#? 
0?????????R@
? ?
J__inference_res_block_1_layer_call_and_return_conditional_losses_196695295?uvwxyz{|}~?????????<
5?2
(?%
inputs?????????R@
p 

 
? "-?*
#? 
0?????????R@
? ?
/__inference_res_block_1_layer_call_fn_196692647?uvwxyz{|}~???????G?D
=?:
0?-
conv2d_5_input?????????R@
p

 
? " ??????????R@?
/__inference_res_block_1_layer_call_fn_196692736?uvwxyz{|}~???????G?D
=?:
0?-
conv2d_5_input?????????R@
p 

 
? " ??????????R@?
/__inference_res_block_1_layer_call_fn_196695336~uvwxyz{|}~?????????<
5?2
(?%
inputs?????????R@
p

 
? " ??????????R@?
/__inference_res_block_1_layer_call_fn_196695377~uvwxyz{|}~?????????<
5?2
(?%
inputs?????????R@
p 

 
? " ??????????R@?
'__inference_signature_wrapper_196694169?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
? 
6?3
1
input(?%
input??????????"G?D
B
feature_linear0?-
feature_linear?????????R