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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??&
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
trainable_variables
regularization_losses
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
trainable_variables
regularization_losses
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
)trainable_variables
*regularization_losses
+	variables
,	keras_api
4
#-_self_saveable_object_factories
.	keras_api
w
#/_self_saveable_object_factories
0trainable_variables
1regularization_losses
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
=trainable_variables
>regularization_losses
?	variables
@	keras_api
4
#A_self_saveable_object_factories
B	keras_api
w
#C_self_saveable_object_factories
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
?
Hlayer_with_weights-0
Hlayer-0
Ilayer_with_weights-1
Ilayer-1
Jlayer-2
#K_self_saveable_object_factories
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
4
#P_self_saveable_object_factories
Q	keras_api
w
#R_self_saveable_object_factories
Strainable_variables
Tregularization_losses
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
?non_trainable_variables
?metrics
trainable_variables
?layer_metrics
regularization_losses
	variables
 ?layer_regularization_losses
?layers
 
?

Wkernel
Xbias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis
	Ygamma
Zbeta
[moving_mean
\moving_variance
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?

]kernel
^bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis
	_gamma
`beta
amoving_mean
bmoving_variance
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
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
?non_trainable_variables
?metrics
trainable_variables
?layer_metrics
regularization_losses
	variables
 ?layer_regularization_losses
?layers
?

ckernel
dbias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis
	egamma
fbeta
gmoving_mean
hmoving_variance
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?

ikernel
jbias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis
	kgamma
lbeta
mmoving_mean
nmoving_variance
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?

okernel
pbias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis
	qgamma
rbeta
smoving_mean
tmoving_variance
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
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
?non_trainable_variables
?metrics
)trainable_variables
?layer_metrics
*regularization_losses
+	variables
 ?layer_regularization_losses
?layers
 
 
 
 
 
 
?
?non_trainable_variables
?metrics
0trainable_variables
?layer_metrics
1regularization_losses
2	variables
 ?layer_regularization_losses
?layers
?

ukernel
vbias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis
	wgamma
xbeta
ymoving_mean
zmoving_variance
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?

{kernel
|bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis
	}gamma
~beta
moving_mean
?moving_variance
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
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
?non_trainable_variables
?metrics
=trainable_variables
?layer_metrics
>regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 
 
 
?
?non_trainable_variables
?metrics
Dtrainable_variables
?layer_metrics
Eregularization_losses
F	variables
 ?layer_regularization_losses
?layers
?
?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
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
?non_trainable_variables
?metrics
Ltrainable_variables
?layer_metrics
Mregularization_losses
N	variables
 ?layer_regularization_losses
?layers
 
 
 
 
 
 
?
?non_trainable_variables
?metrics
Strainable_variables
?layer_metrics
Tregularization_losses
U	variables
 ?layer_regularization_losses
?layers
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
 

W0
X1
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 

Y0
Z1
[2
\3
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 

]0
^1
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 

_0
`1
a2
b3
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
*
0
1
2
3
4
5
 
 
 

c0
d1
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 

e0
f1
g2
h3
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 

i0
j1
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 

k0
l1
m2
n3
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 

o0
p1
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 

q0
r1
s2
t3
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 

w0
x1
y2
z3
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 

{0
|1
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 

}0
~1
2
?3
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 

?0
?1
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 
 
?0
?1
?2
?3
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 
 
?0
?1
?2
?3
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
 
 
 
 
?
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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

H0
I1
J2
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
GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_9739111
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_9741964
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
GPU 2J 8? *,
f'R%
#__inference__traced_restore_9742136??$
?
?
7__inference_batch_normalization_4_layer_call_fn_9741115

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_97365652
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
?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_9736404

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
?.
?
H__inference_res_block_1_layer_call_and_return_conditional_losses_9737550

inputs
conv2d_5_9737505
conv2d_5_9737507!
batch_normalization_5_9737510!
batch_normalization_5_9737512!
batch_normalization_5_9737514!
batch_normalization_5_9737516
conv2d_6_9737520
conv2d_6_9737522!
batch_normalization_6_9737525!
batch_normalization_6_9737527!
batch_normalization_6_9737529!
batch_normalization_6_9737531
conv2d_7_9737535
conv2d_7_9737537!
batch_normalization_7_9737540!
batch_normalization_7_9737542!
batch_normalization_7_9737544!
batch_normalization_7_9737546
identity??-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_9737505conv2d_5_9737507*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_97371432"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_9737510batch_normalization_5_9737512batch_normalization_5_9737514batch_normalization_5_9737516*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_97371762/
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
GPU 2J 8? *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_97372352
activation_4/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_9737520conv2d_6_9737522*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_97372532"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_9737525batch_normalization_6_9737527batch_normalization_6_9737529batch_normalization_6_9737531*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_97372862/
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
GPU 2J 8? *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_97373452
activation_5/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_9737535conv2d_7_9737537*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_97373632"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_9737540batch_normalization_7_9737542batch_normalization_7_9737544batch_normalization_7_9737546*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_97373962/
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
?
?
D__inference_feature_layer_call_and_return_conditional_losses_9737970

inputs
conv2d_8_9737954
conv2d_8_9737956!
batch_normalization_8_9737959!
batch_normalization_8_9737961!
batch_normalization_8_9737963!
batch_normalization_8_9737965
identity??-batch_normalization_8/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_9737954conv2d_8_9737956*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_97377922"
 conv2d_8/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_9737959batch_normalization_8_9737961batch_normalization_8_9737963batch_normalization_8_9737965*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_97378432/
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
GPU 2J 8? *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_97378842
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
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9741089

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
7__inference_batch_normalization_6_layer_call_fn_9741398

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_97369872
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9737304

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
)__inference_feature_layer_call_fn_9737949
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
GPU 2J 8? *M
fHRF
D__inference_feature_layer_call_and_return_conditional_losses_97379342
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
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9737825

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
?9
?
B__inference_model_layer_call_and_return_conditional_losses_9738400	
input
input_conv_9738047
input_conv_9738049
input_conv_9738051
input_conv_9738053
input_conv_9738055
input_conv_9738057
input_conv_9738059
input_conv_9738061
input_conv_9738063
input_conv_9738065
input_conv_9738067
input_conv_9738069
res_block_0_9738154
res_block_0_9738156
res_block_0_9738158
res_block_0_9738160
res_block_0_9738162
res_block_0_9738164
res_block_0_9738166
res_block_0_9738168
res_block_0_9738170
res_block_0_9738172
res_block_0_9738174
res_block_0_9738176
res_block_0_9738178
res_block_0_9738180
res_block_0_9738182
res_block_0_9738184
res_block_0_9738186
res_block_0_9738188
res_block_1_9738287
res_block_1_9738289
res_block_1_9738291
res_block_1_9738293
res_block_1_9738295
res_block_1_9738297
res_block_1_9738299
res_block_1_9738301
res_block_1_9738303
res_block_1_9738305
res_block_1_9738307
res_block_1_9738309
res_block_1_9738311
res_block_1_9738313
res_block_1_9738315
res_block_1_9738317
res_block_1_9738319
res_block_1_9738321
feature_9738372
feature_9738374
feature_9738376
feature_9738378
feature_9738380
feature_9738382
identity??feature/StatefulPartitionedCall?"input_conv/StatefulPartitionedCall?#res_block_0/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputinput_conv_9738047input_conv_9738049input_conv_9738051input_conv_9738053input_conv_9738055input_conv_9738057input_conv_9738059input_conv_9738061input_conv_9738063input_conv_9738065input_conv_9738067input_conv_9738069*
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
GPU 2J 8? *P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_97358902$
"input_conv/StatefulPartitionedCall?
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_9738154res_block_0_9738156res_block_0_9738158res_block_0_9738160res_block_0_9738162res_block_0_9738164res_block_0_9738166res_block_0_9738168res_block_0_9738170res_block_0_9738172res_block_0_9738174res_block_0_9738176res_block_0_9738178res_block_0_9738180res_block_0_9738182res_block_0_9738184res_block_0_9738186res_block_0_9738188*
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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_0_layer_call_and_return_conditional_losses_97367012%
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
GPU 2J 8? *L
fGRE
C__inference_relu_0_layer_call_and_return_conditional_losses_97381972
relu_0/PartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_9738287res_block_1_9738289res_block_1_9738291res_block_1_9738293res_block_1_9738295res_block_1_9738297res_block_1_9738299res_block_1_9738301res_block_1_9738303res_block_1_9738305res_block_1_9738307res_block_1_9738309res_block_1_9738311res_block_1_9738313res_block_1_9738315res_block_1_9738317res_block_1_9738319res_block_1_9738321*
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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_1_layer_call_and_return_conditional_losses_97375502%
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
GPU 2J 8? *L
fGRE
C__inference_relu_1_layer_call_and_return_conditional_losses_97383302
relu_1/PartitionedCall?
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_9738372feature_9738374feature_9738376feature_9738378feature_9738380feature_9738382*
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
GPU 2J 8? *M
fHRF
D__inference_feature_layer_call_and_return_conditional_losses_97379342!
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
GPU 2J 8? *T
fORM
K__inference_feature_linear_layer_call_and_return_conditional_losses_97383912 
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
?
?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9741385

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
7__inference_batch_normalization_1_layer_call_fn_9740643

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_97355522
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9736455

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
7__inference_batch_normalization_3_layer_call_fn_9741011

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_97361382
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9741133

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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9735483

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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9740477

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
E__inference_conv2d_6_layer_call_and_return_conditional_losses_9741340

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
?
H
,__inference_activation_layer_call_fn_9740575

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
GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_97357002
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
?
?
'__inference_model_layer_call_fn_9738996	
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
GPU 2J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_97388852
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
?
?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9740674

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
I__inference_activation_5_layer_call_and_return_conditional_losses_9737345

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
7__inference_batch_normalization_4_layer_call_fn_9741164

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_97362382
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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9740612

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
E__inference_conv2d_3_layer_call_and_return_conditional_losses_9740891

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
?
?
7__inference_batch_normalization_7_layer_call_fn_9741613

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_97370872
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
?	
?
E__inference_conv2d_7_layer_call_and_return_conditional_losses_9737363

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
)__inference_feature_layer_call_fn_9740413

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
GPU 2J 8? *M
fHRF
D__inference_feature_layer_call_and_return_conditional_losses_97379702
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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9741663

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
?
?
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9740936

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
?!
?
G__inference_input_conv_layer_call_and_return_conditional_losses_9735890

inputs
conv2d_9735859
conv2d_9735861
batch_normalization_9735864
batch_normalization_9735866
batch_normalization_9735868
batch_normalization_9735870
conv2d_1_9735874
conv2d_1_9735876!
batch_normalization_1_9735879!
batch_normalization_1_9735881!
batch_normalization_1_9735883!
batch_normalization_1_9735885
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9735859conv2d_9735861*
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
GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_97356082 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_9735864batch_normalization_9735866batch_normalization_9735868batch_normalization_9735870*
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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_97356412-
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
GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_97357002
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_9735874conv2d_1_9735876*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_97357182"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_9735879batch_normalization_1_9735881batch_normalization_1_9735883batch_normalization_1_9735885*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_97357512/
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
GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_97358102
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
?
?
-__inference_res_block_1_layer_call_fn_9740278

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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_1_layer_call_and_return_conditional_losses_97375502
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
e
I__inference_activation_1_layer_call_and_return_conditional_losses_9735810

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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9741276

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
7__inference_batch_normalization_5_layer_call_fn_9741320

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_97371942
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
?

*__inference_conv2d_8_layer_call_fn_9741645

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
GPU 2J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_97377922
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
?
}
(__inference_conv2d_layer_call_fn_9740441

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
GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_97356082
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
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9741151

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
?
D__inference_feature_layer_call_and_return_conditional_losses_9740354

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
?
?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9741429

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
?
?
'__inference_model_layer_call_fn_9739731

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
GPU 2J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_97388852
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
)__inference_feature_layer_call_fn_9737985
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
GPU 2J 8? *M
fHRF
D__inference_feature_layer_call_and_return_conditional_losses_97379702
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
?
-__inference_res_block_0_layer_call_fn_9736740
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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_0_layer_call_and_return_conditional_losses_97367012
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
??
?8
"__inference__wrapped_model_9735394	
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
?
?
7__inference_batch_normalization_6_layer_call_fn_9741473

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_97373042
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
?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_9740585

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
?
e
I__inference_activation_1_layer_call_and_return_conditional_losses_9740723

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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9740459

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
?
?
7__inference_batch_normalization_3_layer_call_fn_9740962

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_97364552
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
?

*__inference_conv2d_7_layer_call_fn_9741502

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
GPU 2J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_97373632
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
?
e
I__inference_activation_6_layer_call_and_return_conditional_losses_9737884

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
?
?
-__inference_res_block_0_layer_call_fn_9740095

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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_0_layer_call_and_return_conditional_losses_97367902
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
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9737414

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
_
C__inference_relu_1_layer_call_and_return_conditional_losses_9740324

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
D
(__inference_relu_1_layer_call_fn_9740329

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
GPU 2J 8? *L
fGRE
C__inference_relu_1_layer_call_and_return_conditional_losses_97383302
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
?
?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9740630

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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9740918

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
E__inference_conv2d_4_layer_call_and_return_conditional_losses_9741044

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
?
?
7__inference_batch_normalization_2_layer_call_fn_9740809

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_97360692
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9736269

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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9741214

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
_
C__inference_relu_0_layer_call_and_return_conditional_losses_9740100

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
J
.__inference_activation_5_layer_call_fn_9741483

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
GPU 2J 8? *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_97373452
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
?
D
(__inference_relu_0_layer_call_fn_9740105

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
GPU 2J 8? *L
fGRE
C__inference_relu_0_layer_call_and_return_conditional_losses_97381972
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
?.
?
H__inference_res_block_0_layer_call_and_return_conditional_losses_9736602
conv2d_2_input
conv2d_2_9736305
conv2d_2_9736307!
batch_normalization_2_9736372!
batch_normalization_2_9736374!
batch_normalization_2_9736376!
batch_normalization_2_9736378
conv2d_3_9736415
conv2d_3_9736417!
batch_normalization_3_9736482!
batch_normalization_3_9736484!
batch_normalization_3_9736486!
batch_normalization_3_9736488
conv2d_4_9736525
conv2d_4_9736527!
batch_normalization_4_9736592!
batch_normalization_4_9736594!
batch_normalization_4_9736596!
batch_normalization_4_9736598
identity??-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_9736305conv2d_2_9736307*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_97362942"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_9736372batch_normalization_2_9736374batch_normalization_2_9736376batch_normalization_2_9736378*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_97363272/
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
GPU 2J 8? *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_97363862
activation_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_9736415conv2d_3_9736417*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_97364042"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_9736482batch_normalization_3_9736484batch_normalization_3_9736486batch_normalization_3_9736488*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_97364372/
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
GPU 2J 8? *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_97364962
activation_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_9736525conv2d_4_9736527*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_97365142"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_9736592batch_normalization_4_9736594batch_normalization_4_9736596batch_normalization_4_9736598*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_97365472/
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
e
I__inference_activation_3_layer_call_and_return_conditional_losses_9736496

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
7__inference_batch_normalization_1_layer_call_fn_9740718

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_97357692
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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9735452

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
?
?
%__inference_signature_wrapper_9739111	
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
GPU 2J 8? *+
f&R$
"__inference__wrapped_model_97353942
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
?
?
7__inference_batch_normalization_3_layer_call_fn_9740949

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_97364372
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
?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_9737143

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
C__inference_conv2d_layer_call_and_return_conditional_losses_9735608

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
?
?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9737087

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
7__inference_batch_normalization_6_layer_call_fn_9741411

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_97370182
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
7__inference_batch_normalization_8_layer_call_fn_9741707

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_97377672
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
?
?
5__inference_batch_normalization_layer_call_fn_9740565

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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_97354832
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
?
?
7__inference_batch_normalization_5_layer_call_fn_9741307

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_97371762
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
?
?
7__inference_batch_normalization_4_layer_call_fn_9741177

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_97362692
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9736345

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
I__inference_activation_4_layer_call_and_return_conditional_losses_9741325

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
?

*__inference_conv2d_4_layer_call_fn_9741053

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
GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_97365142
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
?V
?
H__inference_res_block_1_layer_call_and_return_conditional_losses_9740237

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
?
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9736069

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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9740998

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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9736238

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
7__inference_batch_normalization_8_layer_call_fn_9741769

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_97378432
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
?
?
)__inference_feature_layer_call_fn_9740396

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
GPU 2J 8? *M
fHRF
D__inference_feature_layer_call_and_return_conditional_losses_97379342
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
?
?
7__inference_batch_normalization_2_layer_call_fn_9740796

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_97360382
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
?	
?
C__inference_conv2d_layer_call_and_return_conditional_losses_9740432

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
_
C__inference_relu_1_layer_call_and_return_conditional_losses_9738330

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
?.
?
H__inference_res_block_1_layer_call_and_return_conditional_losses_9737451
conv2d_5_input
conv2d_5_9737154
conv2d_5_9737156!
batch_normalization_5_9737221!
batch_normalization_5_9737223!
batch_normalization_5_9737225!
batch_normalization_5_9737227
conv2d_6_9737264
conv2d_6_9737266!
batch_normalization_6_9737331!
batch_normalization_6_9737333!
batch_normalization_6_9737335!
batch_normalization_6_9737337
conv2d_7_9737374
conv2d_7_9737376!
batch_normalization_7_9737441!
batch_normalization_7_9737443!
batch_normalization_7_9737445!
batch_normalization_7_9737447
identity??-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_9737154conv2d_5_9737156*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_97371432"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_9737221batch_normalization_5_9737223batch_normalization_5_9737225batch_normalization_5_9737227*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_97371762/
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
GPU 2J 8? *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_97372352
activation_4/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_9737264conv2d_6_9737266*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_97372532"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_9737331batch_normalization_6_9737333batch_normalization_6_9737335batch_normalization_6_9737337*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_97372862/
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
GPU 2J 8? *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_97373452
activation_5/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_9737374conv2d_7_9737376*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_97373632"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_9737441batch_normalization_7_9737443batch_normalization_7_9737445batch_normalization_7_9737447*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_97373962/
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
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9737843

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
?	
?
,__inference_input_conv_layer_call_fn_9739881

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
GPU 2J 8? *P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_97359532
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
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9741582

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9736987

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9740692

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
?9
?
B__inference_model_layer_call_and_return_conditional_losses_9738523	
input
input_conv_9738403
input_conv_9738405
input_conv_9738407
input_conv_9738409
input_conv_9738411
input_conv_9738413
input_conv_9738415
input_conv_9738417
input_conv_9738419
input_conv_9738421
input_conv_9738423
input_conv_9738425
res_block_0_9738428
res_block_0_9738430
res_block_0_9738432
res_block_0_9738434
res_block_0_9738436
res_block_0_9738438
res_block_0_9738440
res_block_0_9738442
res_block_0_9738444
res_block_0_9738446
res_block_0_9738448
res_block_0_9738450
res_block_0_9738452
res_block_0_9738454
res_block_0_9738456
res_block_0_9738458
res_block_0_9738460
res_block_0_9738462
res_block_1_9738467
res_block_1_9738469
res_block_1_9738471
res_block_1_9738473
res_block_1_9738475
res_block_1_9738477
res_block_1_9738479
res_block_1_9738481
res_block_1_9738483
res_block_1_9738485
res_block_1_9738487
res_block_1_9738489
res_block_1_9738491
res_block_1_9738493
res_block_1_9738495
res_block_1_9738497
res_block_1_9738499
res_block_1_9738501
feature_9738506
feature_9738508
feature_9738510
feature_9738512
feature_9738514
feature_9738516
identity??feature/StatefulPartitionedCall?"input_conv/StatefulPartitionedCall?#res_block_0/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputinput_conv_9738403input_conv_9738405input_conv_9738407input_conv_9738409input_conv_9738411input_conv_9738413input_conv_9738415input_conv_9738417input_conv_9738419input_conv_9738421input_conv_9738423input_conv_9738425*
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
GPU 2J 8? *P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_97359532$
"input_conv/StatefulPartitionedCall?
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_9738428res_block_0_9738430res_block_0_9738432res_block_0_9738434res_block_0_9738436res_block_0_9738438res_block_0_9738440res_block_0_9738442res_block_0_9738444res_block_0_9738446res_block_0_9738448res_block_0_9738450res_block_0_9738452res_block_0_9738454res_block_0_9738456res_block_0_9738458res_block_0_9738460res_block_0_9738462*
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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_0_layer_call_and_return_conditional_losses_97367902%
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
GPU 2J 8? *L
fGRE
C__inference_relu_0_layer_call_and_return_conditional_losses_97381972
relu_0/PartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_9738467res_block_1_9738469res_block_1_9738471res_block_1_9738473res_block_1_9738475res_block_1_9738477res_block_1_9738479res_block_1_9738481res_block_1_9738483res_block_1_9738485res_block_1_9738487res_block_1_9738489res_block_1_9738491res_block_1_9738493res_block_1_9738495res_block_1_9738497res_block_1_9738499res_block_1_9738501*
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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_1_layer_call_and_return_conditional_losses_97376392%
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
GPU 2J 8? *L
fGRE
C__inference_relu_1_layer_call_and_return_conditional_losses_97383302
relu_1/PartitionedCall?
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_9738506feature_9738508feature_9738510feature_9738512feature_9738514feature_9738516*
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
GPU 2J 8? *M
fHRF
D__inference_feature_layer_call_and_return_conditional_losses_97379702!
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
GPU 2J 8? *T
fORM
K__inference_feature_linear_layer_call_and_return_conditional_losses_97383912 
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
?
?
7__inference_batch_normalization_6_layer_call_fn_9741460

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_97372862
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
?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_9741187

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
,__inference_input_conv_layer_call_fn_9735917
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
GPU 2J 8? *P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_97358902
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
?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9735641

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
?!
?
G__inference_input_conv_layer_call_and_return_conditional_losses_9735853
conv2d_input
conv2d_9735822
conv2d_9735824
batch_normalization_9735827
batch_normalization_9735829
batch_normalization_9735831
batch_normalization_9735833
conv2d_1_9735837
conv2d_1_9735839!
batch_normalization_1_9735842!
batch_normalization_1_9735844!
batch_normalization_1_9735846!
batch_normalization_1_9735848
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_9735822conv2d_9735824*
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
GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_97356082 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_9735827batch_normalization_9735829batch_normalization_9735831batch_normalization_9735833*
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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_97356592-
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
GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_97357002
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_9735837conv2d_1_9735839*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_97357182"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_9735842batch_normalization_1_9735844batch_normalization_1_9735846batch_normalization_1_9735848*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_97357692/
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
GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_97358102
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
?
?
'__inference_model_layer_call_fn_9738760	
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
GPU 2J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_97386492
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
?	
?
E__inference_conv2d_8_layer_call_and_return_conditional_losses_9737792

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
?
c
G__inference_activation_layer_call_and_return_conditional_losses_9735700

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
?
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9740827

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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9741071

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
I__inference_activation_2_layer_call_and_return_conditional_losses_9736386

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
?9
?
B__inference_model_layer_call_and_return_conditional_losses_9738885

inputs
input_conv_9738765
input_conv_9738767
input_conv_9738769
input_conv_9738771
input_conv_9738773
input_conv_9738775
input_conv_9738777
input_conv_9738779
input_conv_9738781
input_conv_9738783
input_conv_9738785
input_conv_9738787
res_block_0_9738790
res_block_0_9738792
res_block_0_9738794
res_block_0_9738796
res_block_0_9738798
res_block_0_9738800
res_block_0_9738802
res_block_0_9738804
res_block_0_9738806
res_block_0_9738808
res_block_0_9738810
res_block_0_9738812
res_block_0_9738814
res_block_0_9738816
res_block_0_9738818
res_block_0_9738820
res_block_0_9738822
res_block_0_9738824
res_block_1_9738829
res_block_1_9738831
res_block_1_9738833
res_block_1_9738835
res_block_1_9738837
res_block_1_9738839
res_block_1_9738841
res_block_1_9738843
res_block_1_9738845
res_block_1_9738847
res_block_1_9738849
res_block_1_9738851
res_block_1_9738853
res_block_1_9738855
res_block_1_9738857
res_block_1_9738859
res_block_1_9738861
res_block_1_9738863
feature_9738868
feature_9738870
feature_9738872
feature_9738874
feature_9738876
feature_9738878
identity??feature/StatefulPartitionedCall?"input_conv/StatefulPartitionedCall?#res_block_0/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_9738765input_conv_9738767input_conv_9738769input_conv_9738771input_conv_9738773input_conv_9738775input_conv_9738777input_conv_9738779input_conv_9738781input_conv_9738783input_conv_9738785input_conv_9738787*
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
GPU 2J 8? *P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_97359532$
"input_conv/StatefulPartitionedCall?
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_9738790res_block_0_9738792res_block_0_9738794res_block_0_9738796res_block_0_9738798res_block_0_9738800res_block_0_9738802res_block_0_9738804res_block_0_9738806res_block_0_9738808res_block_0_9738810res_block_0_9738812res_block_0_9738814res_block_0_9738816res_block_0_9738818res_block_0_9738820res_block_0_9738822res_block_0_9738824*
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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_0_layer_call_and_return_conditional_losses_97367902%
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
GPU 2J 8? *L
fGRE
C__inference_relu_0_layer_call_and_return_conditional_losses_97381972
relu_0/PartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_9738829res_block_1_9738831res_block_1_9738833res_block_1_9738835res_block_1_9738837res_block_1_9738839res_block_1_9738841res_block_1_9738843res_block_1_9738845res_block_1_9738847res_block_1_9738849res_block_1_9738851res_block_1_9738853res_block_1_9738855res_block_1_9738857res_block_1_9738859res_block_1_9738861res_block_1_9738863*
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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_1_layer_call_and_return_conditional_losses_97376392%
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
GPU 2J 8? *L
fGRE
C__inference_relu_1_layer_call_and_return_conditional_losses_97383302
relu_1/PartitionedCall?
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_9738868feature_9738870feature_9738872feature_9738874feature_9738876feature_9738878*
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
GPU 2J 8? *M
fHRF
D__inference_feature_layer_call_and_return_conditional_losses_97379702!
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
GPU 2J 8? *T
fORM
K__inference_feature_linear_layer_call_and_return_conditional_losses_97383912 
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
?
?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9737194

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
?.
?
H__inference_res_block_1_layer_call_and_return_conditional_losses_9737499
conv2d_5_input
conv2d_5_9737454
conv2d_5_9737456!
batch_normalization_5_9737459!
batch_normalization_5_9737461!
batch_normalization_5_9737463!
batch_normalization_5_9737465
conv2d_6_9737469
conv2d_6_9737471!
batch_normalization_6_9737474!
batch_normalization_6_9737476!
batch_normalization_6_9737478!
batch_normalization_6_9737480
conv2d_7_9737484
conv2d_7_9737486!
batch_normalization_7_9737489!
batch_normalization_7_9737491!
batch_normalization_7_9737493!
batch_normalization_7_9737495
identity??-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_9737454conv2d_5_9737456*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_97371432"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_9737459batch_normalization_5_9737461batch_normalization_5_9737463batch_normalization_5_9737465*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_97371942/
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
GPU 2J 8? *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_97372352
activation_4/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_9737469conv2d_6_9737471*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_97372532"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_9737474batch_normalization_6_9737476batch_normalization_6_9737478batch_normalization_6_9737480*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_97373042/
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
GPU 2J 8? *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_97373452
activation_5/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_9737484conv2d_7_9737486*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_97373632"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_9737489batch_normalization_7_9737491batch_normalization_7_9737493batch_normalization_7_9737495*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_97374142/
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
?
?
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9736138

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
?

*__inference_conv2d_6_layer_call_fn_9741349

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
GPU 2J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_97372532
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
?
?
7__inference_batch_normalization_4_layer_call_fn_9741102

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_97365472
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
?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_9737253

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
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9741520

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
7__inference_batch_normalization_1_layer_call_fn_9740656

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_97355832
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
5__inference_batch_normalization_layer_call_fn_9740490

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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_97356412
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
??
?3
B__inference_model_layer_call_and_return_conditional_losses_9739308

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
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9741743

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
?
g
K__inference_feature_linear_layer_call_and_return_conditional_losses_9740417

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
?
g
K__inference_feature_linear_layer_call_and_return_conditional_losses_9738391

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
?
,__inference_input_conv_layer_call_fn_9739852

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
GPU 2J 8? *P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_97358902
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
?
?
'__inference_model_layer_call_fn_9739618

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
GPU 2J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_97386492
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9737018

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9735769

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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9740845

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
E__inference_conv2d_7_layer_call_and_return_conditional_losses_9741493

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
?
?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9737118

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
I__inference_activation_2_layer_call_and_return_conditional_losses_9740876

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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9736547

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
I__inference_activation_4_layer_call_and_return_conditional_losses_9737235

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
?V
?
H__inference_res_block_1_layer_call_and_return_conditional_losses_9740171

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
?
?
-__inference_res_block_0_layer_call_fn_9736829
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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_0_layer_call_and_return_conditional_losses_97367902
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
?.
?
H__inference_res_block_0_layer_call_and_return_conditional_losses_9736701

inputs
conv2d_2_9736656
conv2d_2_9736658!
batch_normalization_2_9736661!
batch_normalization_2_9736663!
batch_normalization_2_9736665!
batch_normalization_2_9736667
conv2d_3_9736671
conv2d_3_9736673!
batch_normalization_3_9736676!
batch_normalization_3_9736678!
batch_normalization_3_9736680!
batch_normalization_3_9736682
conv2d_4_9736686
conv2d_4_9736688!
batch_normalization_4_9736691!
batch_normalization_4_9736693!
batch_normalization_4_9736695!
batch_normalization_4_9736697
identity??-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_9736656conv2d_2_9736658*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_97362942"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_9736661batch_normalization_2_9736663batch_normalization_2_9736665batch_normalization_2_9736667*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_97363272/
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
GPU 2J 8? *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_97363862
activation_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_9736671conv2d_3_9736673*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_97364042"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_9736676batch_normalization_3_9736678batch_normalization_3_9736680batch_normalization_3_9736682*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_97364372/
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
GPU 2J 8? *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_97364962
activation_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_9736686conv2d_4_9736688*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_97365142"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_9736691batch_normalization_4_9736693batch_normalization_4_9736695batch_normalization_4_9736697*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_97365472/
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
?
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9736038

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9737286

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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9736887

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
?
J
.__inference_activation_4_layer_call_fn_9741330

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
GPU 2J 8? *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_97372352
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
?
?
D__inference_feature_layer_call_and_return_conditional_losses_9737934

inputs
conv2d_8_9737918
conv2d_8_9737920!
batch_normalization_8_9737923!
batch_normalization_8_9737925!
batch_normalization_8_9737927!
batch_normalization_8_9737929
identity??-batch_normalization_8/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_9737918conv2d_8_9737920*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_97377922"
 conv2d_8/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_9737923batch_normalization_8_9737925batch_normalization_8_9737927batch_normalization_8_9737929*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_97378252/
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
GPU 2J 8? *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_97378842
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
?:
?	
G__inference_input_conv_layer_call_and_return_conditional_losses_9739823

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
?
?
5__inference_batch_normalization_layer_call_fn_9740552

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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_97354522
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
?
e
I__inference_activation_6_layer_call_and_return_conditional_losses_9741774

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
?
?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9736327

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
H__inference_res_block_0_layer_call_and_return_conditional_losses_9740013

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
??
?
#__inference__traced_restore_9742136
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
?
?
D__inference_feature_layer_call_and_return_conditional_losses_9740379

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
?	
?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9740738

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
7__inference_batch_normalization_2_layer_call_fn_9740858

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_97363272
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
?
?
7__inference_batch_normalization_7_layer_call_fn_9741626

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_97371182
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9740783

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
7__inference_batch_normalization_7_layer_call_fn_9741551

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_97373962
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9741294

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
E__inference_conv2d_8_layer_call_and_return_conditional_losses_9741636

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
?
e
I__inference_activation_3_layer_call_and_return_conditional_losses_9741029

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
D__inference_feature_layer_call_and_return_conditional_losses_9737912
conv2d_8_input
conv2d_8_9737896
conv2d_8_9737898!
batch_normalization_8_9737901!
batch_normalization_8_9737903!
batch_normalization_8_9737905!
batch_normalization_8_9737907
identity??-batch_normalization_8/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_9737896conv2d_8_9737898*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_97377922"
 conv2d_8/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_9737901batch_normalization_8_9737903batch_normalization_8_9737905batch_normalization_8_9737907*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_97378432/
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
GPU 2J 8? *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_97378842
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9741232

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
?

*__inference_conv2d_2_layer_call_fn_9740747

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
GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_97362942
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9736437

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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9740765

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
,__inference_input_conv_layer_call_fn_9735980
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
GPU 2J 8? *P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_97359532
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
?V
?
H__inference_res_block_0_layer_call_and_return_conditional_losses_9739947

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
?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9735659

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
?
?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9736918

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
?
J
.__inference_activation_1_layer_call_fn_9740728

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
GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_97358102
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
?!
?
G__inference_input_conv_layer_call_and_return_conditional_losses_9735953

inputs
conv2d_9735922
conv2d_9735924
batch_normalization_9735927
batch_normalization_9735929
batch_normalization_9735931
batch_normalization_9735933
conv2d_1_9735937
conv2d_1_9735939!
batch_normalization_1_9735942!
batch_normalization_1_9735944!
batch_normalization_1_9735946!
batch_normalization_1_9735948
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9735922conv2d_9735924*
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
GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_97356082 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_9735927batch_normalization_9735929batch_normalization_9735931batch_normalization_9735933*
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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_97356592-
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
GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_97357002
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_9735937conv2d_1_9735939*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_97357182"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_9735942batch_normalization_1_9735944batch_normalization_1_9735946batch_normalization_1_9735948*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_97357692/
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
GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_97358102
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
?
?
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9740980

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
?
-__inference_res_block_1_layer_call_fn_9737589
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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_1_layer_call_and_return_conditional_losses_97375502
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
?
J
.__inference_activation_3_layer_call_fn_9741034

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
GPU 2J 8? *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_97364962
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
E__inference_conv2d_4_layer_call_and_return_conditional_losses_9736514

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
-__inference_res_block_1_layer_call_fn_9740319

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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_1_layer_call_and_return_conditional_losses_97376392
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
?

*__inference_conv2d_5_layer_call_fn_9741196

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
GPU 2J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_97371432
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
?

*__inference_conv2d_1_layer_call_fn_9740594

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
GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_97357182
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
?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9740521

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
7__inference_batch_normalization_8_layer_call_fn_9741756

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_97378252
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
?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_9735718

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
?.
?
H__inference_res_block_0_layer_call_and_return_conditional_losses_9736650
conv2d_2_input
conv2d_2_9736605
conv2d_2_9736607!
batch_normalization_2_9736610!
batch_normalization_2_9736612!
batch_normalization_2_9736614!
batch_normalization_2_9736616
conv2d_3_9736620
conv2d_3_9736622!
batch_normalization_3_9736625!
batch_normalization_3_9736627!
batch_normalization_3_9736629!
batch_normalization_3_9736631
conv2d_4_9736635
conv2d_4_9736637!
batch_normalization_4_9736640!
batch_normalization_4_9736642!
batch_normalization_4_9736644!
batch_normalization_4_9736646
identity??-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_9736605conv2d_2_9736607*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_97362942"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_9736610batch_normalization_2_9736612batch_normalization_2_9736614batch_normalization_2_9736616*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_97363452/
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
GPU 2J 8? *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_97363862
activation_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_9736620conv2d_3_9736622*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_97364042"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_9736625batch_normalization_3_9736627batch_normalization_3_9736629batch_normalization_3_9736631*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_97364552/
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
GPU 2J 8? *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_97364962
activation_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_9736635conv2d_4_9736637*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_97365142"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_9736640batch_normalization_4_9736642batch_normalization_4_9736644batch_normalization_4_9736646*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_97365652/
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
?.
?
H__inference_res_block_0_layer_call_and_return_conditional_losses_9736790

inputs
conv2d_2_9736745
conv2d_2_9736747!
batch_normalization_2_9736750!
batch_normalization_2_9736752!
batch_normalization_2_9736754!
batch_normalization_2_9736756
conv2d_3_9736760
conv2d_3_9736762!
batch_normalization_3_9736765!
batch_normalization_3_9736767!
batch_normalization_3_9736769!
batch_normalization_3_9736771
conv2d_4_9736775
conv2d_4_9736777!
batch_normalization_4_9736780!
batch_normalization_4_9736782!
batch_normalization_4_9736784!
batch_normalization_4_9736786
identity??-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_9736745conv2d_2_9736747*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_97362942"
 conv2d_2/StatefulPartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_9736750batch_normalization_2_9736752batch_normalization_2_9736754batch_normalization_2_9736756*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_97363452/
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
GPU 2J 8? *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_97363862
activation_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_9736760conv2d_3_9736762*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_97364042"
 conv2d_3/StatefulPartitionedCall?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_9736765batch_normalization_3_9736767batch_normalization_3_9736769batch_normalization_3_9736771*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_97364552/
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
GPU 2J 8? *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_97364962
activation_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_9736775conv2d_4_9736777*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_97365142"
 conv2d_4/StatefulPartitionedCall?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_9736780batch_normalization_4_9736782batch_normalization_4_9736784batch_normalization_4_9736786*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_97365652/
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
?
?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9737176

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
7__inference_batch_normalization_2_layer_call_fn_9740871

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_97363452
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
_
C__inference_relu_0_layer_call_and_return_conditional_losses_9738197

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
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9737396

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
?!
?
G__inference_input_conv_layer_call_and_return_conditional_losses_9735819
conv2d_input
conv2d_9735619
conv2d_9735621
batch_normalization_9735686
batch_normalization_9735688
batch_normalization_9735690
batch_normalization_9735692
conv2d_1_9735729
conv2d_1_9735731!
batch_normalization_1_9735796!
batch_normalization_1_9735798!
batch_normalization_1_9735800!
batch_normalization_1_9735802
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_9735619conv2d_9735621*
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
GPU 2J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_97356082 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_9735686batch_normalization_9735688batch_normalization_9735690batch_normalization_9735692*
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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_97356412-
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
GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_97357002
activation/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_9735729conv2d_1_9735731*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_97357182"
 conv2d_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_9735796batch_normalization_1_9735798batch_normalization_1_9735800batch_normalization_1_9735802*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_97357512/
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
GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_97358102
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
?
?
7__inference_batch_normalization_8_layer_call_fn_9741694

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_97377362
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
?
?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9741725

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
?
J
.__inference_activation_2_layer_call_fn_9740881

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
GPU 2J 8? *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_97363862
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
?
?
D__inference_feature_layer_call_and_return_conditional_losses_9737893
conv2d_8_input
conv2d_8_9737803
conv2d_8_9737805!
batch_normalization_8_9737870!
batch_normalization_8_9737872!
batch_normalization_8_9737874!
batch_normalization_8_9737876
identity??-batch_normalization_8/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_9737803conv2d_8_9737805*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_97377922"
 conv2d_8/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_9737870batch_normalization_8_9737872batch_normalization_8_9737874batch_normalization_8_9737876*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_97378252/
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
GPU 2J 8? *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_97378842
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
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9741600

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
7__inference_batch_normalization_5_layer_call_fn_9741245

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_97368872
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
?.
?
H__inference_res_block_1_layer_call_and_return_conditional_losses_9737639

inputs
conv2d_5_9737594
conv2d_5_9737596!
batch_normalization_5_9737599!
batch_normalization_5_9737601!
batch_normalization_5_9737603!
batch_normalization_5_9737605
conv2d_6_9737609
conv2d_6_9737611!
batch_normalization_6_9737614!
batch_normalization_6_9737616!
batch_normalization_6_9737618!
batch_normalization_6_9737620
conv2d_7_9737624
conv2d_7_9737626!
batch_normalization_7_9737629!
batch_normalization_7_9737631!
batch_normalization_7_9737633!
batch_normalization_7_9737635
identity??-batch_normalization_5/StatefulPartitionedCall?-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_9737594conv2d_5_9737596*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_97371432"
 conv2d_5/StatefulPartitionedCall?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_9737599batch_normalization_5_9737601batch_normalization_5_9737603batch_normalization_5_9737605*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_97371942/
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
GPU 2J 8? *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_97372352
activation_4/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_9737609conv2d_6_9737611*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_97372532"
 conv2d_6/StatefulPartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_9737614batch_normalization_6_9737616batch_normalization_6_9737618batch_normalization_6_9737620*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_97373042/
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
GPU 2J 8? *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_97373452
activation_5/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_9737624conv2d_7_9737626*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_97373632"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_9737629batch_normalization_7_9737631batch_normalization_7_9737633batch_normalization_7_9737635*
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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_97374142/
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
L
0__inference_feature_linear_layer_call_fn_9740422

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
GPU 2J 8? *T
fORM
K__inference_feature_linear_layer_call_and_return_conditional_losses_97383912
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9741367

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
?9
?
B__inference_model_layer_call_and_return_conditional_losses_9738649

inputs
input_conv_9738529
input_conv_9738531
input_conv_9738533
input_conv_9738535
input_conv_9738537
input_conv_9738539
input_conv_9738541
input_conv_9738543
input_conv_9738545
input_conv_9738547
input_conv_9738549
input_conv_9738551
res_block_0_9738554
res_block_0_9738556
res_block_0_9738558
res_block_0_9738560
res_block_0_9738562
res_block_0_9738564
res_block_0_9738566
res_block_0_9738568
res_block_0_9738570
res_block_0_9738572
res_block_0_9738574
res_block_0_9738576
res_block_0_9738578
res_block_0_9738580
res_block_0_9738582
res_block_0_9738584
res_block_0_9738586
res_block_0_9738588
res_block_1_9738593
res_block_1_9738595
res_block_1_9738597
res_block_1_9738599
res_block_1_9738601
res_block_1_9738603
res_block_1_9738605
res_block_1_9738607
res_block_1_9738609
res_block_1_9738611
res_block_1_9738613
res_block_1_9738615
res_block_1_9738617
res_block_1_9738619
res_block_1_9738621
res_block_1_9738623
res_block_1_9738625
res_block_1_9738627
feature_9738632
feature_9738634
feature_9738636
feature_9738638
feature_9738640
feature_9738642
identity??feature/StatefulPartitionedCall?"input_conv/StatefulPartitionedCall?#res_block_0/StatefulPartitionedCall?#res_block_1/StatefulPartitionedCall?
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_9738529input_conv_9738531input_conv_9738533input_conv_9738535input_conv_9738537input_conv_9738539input_conv_9738541input_conv_9738543input_conv_9738545input_conv_9738547input_conv_9738549input_conv_9738551*
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
GPU 2J 8? *P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_97358902$
"input_conv/StatefulPartitionedCall?
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_9738554res_block_0_9738556res_block_0_9738558res_block_0_9738560res_block_0_9738562res_block_0_9738564res_block_0_9738566res_block_0_9738568res_block_0_9738570res_block_0_9738572res_block_0_9738574res_block_0_9738576res_block_0_9738578res_block_0_9738580res_block_0_9738582res_block_0_9738584res_block_0_9738586res_block_0_9738588*
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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_0_layer_call_and_return_conditional_losses_97367012%
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
GPU 2J 8? *L
fGRE
C__inference_relu_0_layer_call_and_return_conditional_losses_97381972
relu_0/PartitionedCall?
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_9738593res_block_1_9738595res_block_1_9738597res_block_1_9738599res_block_1_9738601res_block_1_9738603res_block_1_9738605res_block_1_9738607res_block_1_9738609res_block_1_9738611res_block_1_9738613res_block_1_9738615res_block_1_9738617res_block_1_9738619res_block_1_9738621res_block_1_9738623res_block_1_9738625res_block_1_9738627*
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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_1_layer_call_and_return_conditional_losses_97375502%
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
GPU 2J 8? *L
fGRE
C__inference_relu_1_layer_call_and_return_conditional_losses_97383302
relu_1/PartitionedCall?
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_9738632feature_9738634feature_9738636feature_9738638feature_9738640feature_9738642*
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
GPU 2J 8? *M
fHRF
D__inference_feature_layer_call_and_return_conditional_losses_97379342!
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
GPU 2J 8? *T
fORM
K__inference_feature_linear_layer_call_and_return_conditional_losses_97383912 
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
??
?3
B__inference_model_layer_call_and_return_conditional_losses_9739505

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
?:
?	
G__inference_input_conv_layer_call_and_return_conditional_losses_9739777

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
?
?
7__inference_batch_normalization_1_layer_call_fn_9740705

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_97357512
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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9737736

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9735583

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
-__inference_res_block_0_layer_call_fn_9740054

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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_0_layer_call_and_return_conditional_losses_97367012
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
?
?
5__inference_batch_normalization_layer_call_fn_9740503

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
GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_97356592
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
?
J
.__inference_activation_6_layer_call_fn_9741779

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
GPU 2J 8? *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_97378842
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
?
?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9736565

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
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9740539

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
?	
?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9736294

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
?
?
7__inference_batch_normalization_5_layer_call_fn_9741258

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_97369182
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9741447

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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9735552

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
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9741538

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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9741681

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
?
c
G__inference_activation_layer_call_and_return_conditional_losses_9740570

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
?
?
7__inference_batch_normalization_7_layer_call_fn_9741564

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_97374142
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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9735751

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
7__inference_batch_normalization_3_layer_call_fn_9741024

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
GPU 2J 8? *[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_97361692
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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9737767

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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9736169

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
I__inference_activation_5_layer_call_and_return_conditional_losses_9741478

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
?g
?
 __inference__traced_save_9741964
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
?
?
-__inference_res_block_1_layer_call_fn_9737678
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
GPU 2J 8? *Q
fLRJ
H__inference_res_block_1_layer_call_and_return_conditional_losses_97376392
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
?

*__inference_conv2d_3_layer_call_fn_9740900

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
GPU 2J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_97364042
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
trainable_variables
regularization_losses
	variables
	keras_api
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"??
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
trainable_variables
regularization_losses
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
)trainable_variables
*regularization_losses
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
0trainable_variables
1regularization_losses
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
=trainable_variables
>regularization_losses
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
Dtrainable_variables
Eregularization_losses
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
Ltrainable_variables
Mregularization_losses
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
Strainable_variables
Tregularization_losses
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
?non_trainable_variables
?metrics
trainable_variables
?layer_metrics
regularization_losses
	variables
 ?layer_regularization_losses
?layers
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
?trainable_variables
?regularization_losses
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
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 168, 128]}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}
?


]kernel
^bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
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
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
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
?non_trainable_variables
?metrics
trainable_variables
?layer_metrics
regularization_losses
	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?


ckernel
dbias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
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
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}
?


ikernel
jbias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
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
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}
?


okernel
pbias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
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
?trainable_variables
?regularization_losses
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
?non_trainable_variables
?metrics
)trainable_variables
?layer_metrics
*regularization_losses
+	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
0trainable_variables
?layer_metrics
1regularization_losses
2	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?


ukernel
vbias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
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
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}
?


{kernel
|bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
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
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_5", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}
?

?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
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
?trainable_variables
?regularization_losses
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
?non_trainable_variables
?metrics
=trainable_variables
?layer_metrics
>regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
Dtrainable_variables
?layer_metrics
Eregularization_losses
F	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
	?bias
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
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
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 8]}}
?
$?_self_saveable_object_factories
?trainable_variables
?regularization_losses
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
?non_trainable_variables
?metrics
Ltrainable_variables
?layer_metrics
Mregularization_losses
N	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
Strainable_variables
?layer_metrics
Tregularization_losses
U	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
J
0
1
2
3
4
5"
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
trackable_dict_wrapper
 "
trackable_list_wrapper
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
trackable_dict_wrapper
 "
trackable_list_wrapper
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
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
?non_trainable_variables
?metrics
?trainable_variables
?layer_metrics
?regularization_losses
?	variables
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
trackable_dict_wrapper
 "
trackable_list_wrapper
5
H0
I1
J2"
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
.
W0
X1"
trackable_list_wrapper
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
.
]0
^1"
trackable_list_wrapper
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
.
c0
d1"
trackable_list_wrapper
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
.
i0
j1"
trackable_list_wrapper
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
.
o0
p1"
trackable_list_wrapper
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
.
{0
|1"
trackable_list_wrapper
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
0
?0
?1"
trackable_list_wrapper
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
?2?
"__inference__wrapped_model_9735394?
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
?2?
'__inference_model_layer_call_fn_9738760
'__inference_model_layer_call_fn_9739731
'__inference_model_layer_call_fn_9739618
'__inference_model_layer_call_fn_9738996?
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
B__inference_model_layer_call_and_return_conditional_losses_9739505
B__inference_model_layer_call_and_return_conditional_losses_9738400
B__inference_model_layer_call_and_return_conditional_losses_9739308
B__inference_model_layer_call_and_return_conditional_losses_9738523?
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
,__inference_input_conv_layer_call_fn_9739881
,__inference_input_conv_layer_call_fn_9739852
,__inference_input_conv_layer_call_fn_9735980
,__inference_input_conv_layer_call_fn_9735917?
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
G__inference_input_conv_layer_call_and_return_conditional_losses_9735819
G__inference_input_conv_layer_call_and_return_conditional_losses_9739823
G__inference_input_conv_layer_call_and_return_conditional_losses_9739777
G__inference_input_conv_layer_call_and_return_conditional_losses_9735853?
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
?2?
-__inference_res_block_0_layer_call_fn_9740054
-__inference_res_block_0_layer_call_fn_9736829
-__inference_res_block_0_layer_call_fn_9740095
-__inference_res_block_0_layer_call_fn_9736740?
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
H__inference_res_block_0_layer_call_and_return_conditional_losses_9739947
H__inference_res_block_0_layer_call_and_return_conditional_losses_9736650
H__inference_res_block_0_layer_call_and_return_conditional_losses_9740013
H__inference_res_block_0_layer_call_and_return_conditional_losses_9736602?
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
(__inference_relu_0_layer_call_fn_9740105?
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
C__inference_relu_0_layer_call_and_return_conditional_losses_9740100?
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
?2?
-__inference_res_block_1_layer_call_fn_9740278
-__inference_res_block_1_layer_call_fn_9737589
-__inference_res_block_1_layer_call_fn_9737678
-__inference_res_block_1_layer_call_fn_9740319?
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
H__inference_res_block_1_layer_call_and_return_conditional_losses_9737499
H__inference_res_block_1_layer_call_and_return_conditional_losses_9740237
H__inference_res_block_1_layer_call_and_return_conditional_losses_9740171
H__inference_res_block_1_layer_call_and_return_conditional_losses_9737451?
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
(__inference_relu_1_layer_call_fn_9740329?
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
C__inference_relu_1_layer_call_and_return_conditional_losses_9740324?
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
)__inference_feature_layer_call_fn_9740396
)__inference_feature_layer_call_fn_9737949
)__inference_feature_layer_call_fn_9740413
)__inference_feature_layer_call_fn_9737985?
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
D__inference_feature_layer_call_and_return_conditional_losses_9737893
D__inference_feature_layer_call_and_return_conditional_losses_9740354
D__inference_feature_layer_call_and_return_conditional_losses_9740379
D__inference_feature_layer_call_and_return_conditional_losses_9737912?
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
0__inference_feature_linear_layer_call_fn_9740422?
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
K__inference_feature_linear_layer_call_and_return_conditional_losses_9740417?
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
%__inference_signature_wrapper_9739111input"?
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
(__inference_conv2d_layer_call_fn_9740441?
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
C__inference_conv2d_layer_call_and_return_conditional_losses_9740432?
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
5__inference_batch_normalization_layer_call_fn_9740565
5__inference_batch_normalization_layer_call_fn_9740503
5__inference_batch_normalization_layer_call_fn_9740552
5__inference_batch_normalization_layer_call_fn_9740490?
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
?2?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9740539
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9740459
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9740477
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9740521?
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
,__inference_activation_layer_call_fn_9740575?
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
G__inference_activation_layer_call_and_return_conditional_losses_9740570?
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
*__inference_conv2d_1_layer_call_fn_9740594?
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
E__inference_conv2d_1_layer_call_and_return_conditional_losses_9740585?
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
7__inference_batch_normalization_1_layer_call_fn_9740705
7__inference_batch_normalization_1_layer_call_fn_9740656
7__inference_batch_normalization_1_layer_call_fn_9740718
7__inference_batch_normalization_1_layer_call_fn_9740643?
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
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9740692
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9740630
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9740612
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9740674?
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
.__inference_activation_1_layer_call_fn_9740728?
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
I__inference_activation_1_layer_call_and_return_conditional_losses_9740723?
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
*__inference_conv2d_2_layer_call_fn_9740747?
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
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9740738?
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
7__inference_batch_normalization_2_layer_call_fn_9740871
7__inference_batch_normalization_2_layer_call_fn_9740858
7__inference_batch_normalization_2_layer_call_fn_9740796
7__inference_batch_normalization_2_layer_call_fn_9740809?
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
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9740783
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9740827
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9740765
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9740845?
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
.__inference_activation_2_layer_call_fn_9740881?
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
I__inference_activation_2_layer_call_and_return_conditional_losses_9740876?
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
*__inference_conv2d_3_layer_call_fn_9740900?
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
E__inference_conv2d_3_layer_call_and_return_conditional_losses_9740891?
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
7__inference_batch_normalization_3_layer_call_fn_9740962
7__inference_batch_normalization_3_layer_call_fn_9741011
7__inference_batch_normalization_3_layer_call_fn_9740949
7__inference_batch_normalization_3_layer_call_fn_9741024?
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
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9740936
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9740980
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9740998
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9740918?
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
.__inference_activation_3_layer_call_fn_9741034?
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
I__inference_activation_3_layer_call_and_return_conditional_losses_9741029?
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
*__inference_conv2d_4_layer_call_fn_9741053?
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
E__inference_conv2d_4_layer_call_and_return_conditional_losses_9741044?
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
7__inference_batch_normalization_4_layer_call_fn_9741177
7__inference_batch_normalization_4_layer_call_fn_9741102
7__inference_batch_normalization_4_layer_call_fn_9741115
7__inference_batch_normalization_4_layer_call_fn_9741164?
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9741071
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9741133
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9741151
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9741089?
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
*__inference_conv2d_5_layer_call_fn_9741196?
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
E__inference_conv2d_5_layer_call_and_return_conditional_losses_9741187?
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
7__inference_batch_normalization_5_layer_call_fn_9741258
7__inference_batch_normalization_5_layer_call_fn_9741320
7__inference_batch_normalization_5_layer_call_fn_9741245
7__inference_batch_normalization_5_layer_call_fn_9741307?
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9741276
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9741232
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9741214
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9741294?
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
.__inference_activation_4_layer_call_fn_9741330?
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
I__inference_activation_4_layer_call_and_return_conditional_losses_9741325?
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
*__inference_conv2d_6_layer_call_fn_9741349?
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
E__inference_conv2d_6_layer_call_and_return_conditional_losses_9741340?
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
7__inference_batch_normalization_6_layer_call_fn_9741411
7__inference_batch_normalization_6_layer_call_fn_9741473
7__inference_batch_normalization_6_layer_call_fn_9741398
7__inference_batch_normalization_6_layer_call_fn_9741460?
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9741385
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9741429
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9741367
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9741447?
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
.__inference_activation_5_layer_call_fn_9741483?
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
I__inference_activation_5_layer_call_and_return_conditional_losses_9741478?
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
*__inference_conv2d_7_layer_call_fn_9741502?
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
E__inference_conv2d_7_layer_call_and_return_conditional_losses_9741493?
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
7__inference_batch_normalization_7_layer_call_fn_9741613
7__inference_batch_normalization_7_layer_call_fn_9741551
7__inference_batch_normalization_7_layer_call_fn_9741564
7__inference_batch_normalization_7_layer_call_fn_9741626?
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
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9741582
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9741520
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9741538
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9741600?
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
*__inference_conv2d_8_layer_call_fn_9741645?
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
E__inference_conv2d_8_layer_call_and_return_conditional_losses_9741636?
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
7__inference_batch_normalization_8_layer_call_fn_9741694
7__inference_batch_normalization_8_layer_call_fn_9741756
7__inference_batch_normalization_8_layer_call_fn_9741769
7__inference_batch_normalization_8_layer_call_fn_9741707?
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
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9741743
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9741725
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9741663
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9741681?
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
.__inference_activation_6_layer_call_fn_9741779?
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
I__inference_activation_6_layer_call_and_return_conditional_losses_9741774?
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
"__inference__wrapped_model_9735394?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????7?4
-?*
(?%
input??????????
? "G?D
B
feature_linear0?-
feature_linear?????????R?
I__inference_activation_1_layer_call_and_return_conditional_losses_9740723h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
.__inference_activation_1_layer_call_fn_9740728[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
I__inference_activation_2_layer_call_and_return_conditional_losses_9740876h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
.__inference_activation_2_layer_call_fn_9740881[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
I__inference_activation_3_layer_call_and_return_conditional_losses_9741029h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
.__inference_activation_3_layer_call_fn_9741034[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
I__inference_activation_4_layer_call_and_return_conditional_losses_9741325h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
.__inference_activation_4_layer_call_fn_9741330[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
I__inference_activation_5_layer_call_and_return_conditional_losses_9741478h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
.__inference_activation_5_layer_call_fn_9741483[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
I__inference_activation_6_layer_call_and_return_conditional_losses_9741774h7?4
-?*
(?%
inputs?????????R
? "-?*
#? 
0?????????R
? ?
.__inference_activation_6_layer_call_fn_9741779[7?4
-?*
(?%
inputs?????????R
? " ??????????R?
G__inference_activation_layer_call_and_return_conditional_losses_9740570l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
,__inference_activation_layer_call_fn_9740575_9?6
/?,
*?'
inputs???????????
? ""?????????????
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9740612?_`abM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9740630?_`abM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9740674r_`ab;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9740692r_`ab;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
7__inference_batch_normalization_1_layer_call_fn_9740643?_`abM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
7__inference_batch_normalization_1_layer_call_fn_9740656?_`abM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
7__inference_batch_normalization_1_layer_call_fn_9740705e_`ab;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
7__inference_batch_normalization_1_layer_call_fn_9740718e_`ab;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9740765?efghM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9740783?efghM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9740827refgh;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9740845refgh;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
7__inference_batch_normalization_2_layer_call_fn_9740796?efghM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
7__inference_batch_normalization_2_layer_call_fn_9740809?efghM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
7__inference_batch_normalization_2_layer_call_fn_9740858eefgh;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
7__inference_batch_normalization_2_layer_call_fn_9740871eefgh;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9740918rklmn;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9740936rklmn;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9740980?klmnM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_9740998?klmnM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
7__inference_batch_normalization_3_layer_call_fn_9740949eklmn;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
7__inference_batch_normalization_3_layer_call_fn_9740962eklmn;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
7__inference_batch_normalization_3_layer_call_fn_9741011?klmnM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
7__inference_batch_normalization_3_layer_call_fn_9741024?klmnM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9741071rqrst;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9741089rqrst;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9741133?qrstM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_9741151?qrstM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
7__inference_batch_normalization_4_layer_call_fn_9741102eqrst;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
7__inference_batch_normalization_4_layer_call_fn_9741115eqrst;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
7__inference_batch_normalization_4_layer_call_fn_9741164?qrstM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
7__inference_batch_normalization_4_layer_call_fn_9741177?qrstM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9741214?wxyzM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9741232?wxyzM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9741276rwxyz;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_9741294rwxyz;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
7__inference_batch_normalization_5_layer_call_fn_9741245?wxyzM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
7__inference_batch_normalization_5_layer_call_fn_9741258?wxyzM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
7__inference_batch_normalization_5_layer_call_fn_9741307ewxyz;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
7__inference_batch_normalization_5_layer_call_fn_9741320ewxyz;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9741367?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9741385?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9741429s}~?;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_9741447s}~?;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
7__inference_batch_normalization_6_layer_call_fn_9741398?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
7__inference_batch_normalization_6_layer_call_fn_9741411?}~?M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
7__inference_batch_normalization_6_layer_call_fn_9741460f}~?;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
7__inference_batch_normalization_6_layer_call_fn_9741473f}~?;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9741520v????;?8
1?.
(?%
inputs?????????R@
p
? "-?*
#? 
0?????????R@
? ?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9741538v????;?8
1?.
(?%
inputs?????????R@
p 
? "-?*
#? 
0?????????R@
? ?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9741582?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_9741600?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
7__inference_batch_normalization_7_layer_call_fn_9741551i????;?8
1?.
(?%
inputs?????????R@
p
? " ??????????R@?
7__inference_batch_normalization_7_layer_call_fn_9741564i????;?8
1?.
(?%
inputs?????????R@
p 
? " ??????????R@?
7__inference_batch_normalization_7_layer_call_fn_9741613?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
7__inference_batch_normalization_7_layer_call_fn_9741626?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9741663?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9741681?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9741725v????;?8
1?.
(?%
inputs?????????R
p
? "-?*
#? 
0?????????R
? ?
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_9741743v????;?8
1?.
(?%
inputs?????????R
p 
? "-?*
#? 
0?????????R
? ?
7__inference_batch_normalization_8_layer_call_fn_9741694?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
7__inference_batch_normalization_8_layer_call_fn_9741707?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
7__inference_batch_normalization_8_layer_call_fn_9741756i????;?8
1?.
(?%
inputs?????????R
p
? " ??????????R?
7__inference_batch_normalization_8_layer_call_fn_9741769i????;?8
1?.
(?%
inputs?????????R
p 
? " ??????????R?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9740459vYZ[\=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9740477vYZ[\=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9740521?YZ[\N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9740539?YZ[\N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
5__inference_batch_normalization_layer_call_fn_9740490iYZ[\=?:
3?0
*?'
inputs???????????
p
? ""?????????????
5__inference_batch_normalization_layer_call_fn_9740503iYZ[\=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
5__inference_batch_normalization_layer_call_fn_9740552?YZ[\N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
5__inference_batch_normalization_layer_call_fn_9740565?YZ[\N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
E__inference_conv2d_1_layer_call_and_return_conditional_losses_9740585n]^9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????R@
? ?
*__inference_conv2d_1_layer_call_fn_9740594a]^9?6
/?,
*?'
inputs???????????
? " ??????????R@?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_9740738lcd7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
*__inference_conv2d_2_layer_call_fn_9740747_cd7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_9740891lij7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
*__inference_conv2d_3_layer_call_fn_9740900_ij7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_9741044lop7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
*__inference_conv2d_4_layer_call_fn_9741053_op7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_9741187luv7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
*__inference_conv2d_5_layer_call_fn_9741196_uv7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_9741340l{|7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
*__inference_conv2d_6_layer_call_fn_9741349_{|7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
E__inference_conv2d_7_layer_call_and_return_conditional_losses_9741493n??7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
*__inference_conv2d_7_layer_call_fn_9741502a??7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
E__inference_conv2d_8_layer_call_and_return_conditional_losses_9741636n??7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R
? ?
*__inference_conv2d_8_layer_call_fn_9741645a??7?4
-?*
(?%
inputs?????????R@
? " ??????????R?
C__inference_conv2d_layer_call_and_return_conditional_losses_9740432oWX8?5
.?+
)?&
inputs??????????
? "/?,
%?"
0???????????
? ?
(__inference_conv2d_layer_call_fn_9740441bWX8?5
.?+
)?&
inputs??????????
? ""?????????????
D__inference_feature_layer_call_and_return_conditional_losses_9737893???????G?D
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
D__inference_feature_layer_call_and_return_conditional_losses_9737912???????G?D
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
D__inference_feature_layer_call_and_return_conditional_losses_9740354~????????<
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
D__inference_feature_layer_call_and_return_conditional_losses_9740379~????????<
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
)__inference_feature_layer_call_fn_9737949y??????G?D
=?:
0?-
conv2d_8_input?????????R@
p

 
? " ??????????R?
)__inference_feature_layer_call_fn_9737985y??????G?D
=?:
0?-
conv2d_8_input?????????R@
p 

 
? " ??????????R?
)__inference_feature_layer_call_fn_9740396q????????<
5?2
(?%
inputs?????????R@
p

 
? " ??????????R?
)__inference_feature_layer_call_fn_9740413q????????<
5?2
(?%
inputs?????????R@
p 

 
? " ??????????R?
K__inference_feature_linear_layer_call_and_return_conditional_losses_9740417h7?4
-?*
(?%
inputs?????????R
? "-?*
#? 
0?????????R
? ?
0__inference_feature_linear_layer_call_fn_9740422[7?4
-?*
(?%
inputs?????????R
? " ??????????R?
G__inference_input_conv_layer_call_and_return_conditional_losses_9735819?WXYZ[\]^_`abF?C
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
G__inference_input_conv_layer_call_and_return_conditional_losses_9735853?WXYZ[\]^_`abF?C
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
G__inference_input_conv_layer_call_and_return_conditional_losses_9739777WXYZ[\]^_`ab@?=
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
G__inference_input_conv_layer_call_and_return_conditional_losses_9739823WXYZ[\]^_`ab@?=
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
,__inference_input_conv_layer_call_fn_9735917xWXYZ[\]^_`abF?C
<?9
/?,
conv2d_input??????????
p

 
? " ??????????R@?
,__inference_input_conv_layer_call_fn_9735980xWXYZ[\]^_`abF?C
<?9
/?,
conv2d_input??????????
p 

 
? " ??????????R@?
,__inference_input_conv_layer_call_fn_9739852rWXYZ[\]^_`ab@?=
6?3
)?&
inputs??????????
p

 
? " ??????????R@?
,__inference_input_conv_layer_call_fn_9739881rWXYZ[\]^_`ab@?=
6?3
)?&
inputs??????????
p 

 
? " ??????????R@?
B__inference_model_layer_call_and_return_conditional_losses_9738400?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????????<
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
B__inference_model_layer_call_and_return_conditional_losses_9738523?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????????<
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
B__inference_model_layer_call_and_return_conditional_losses_9739308?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
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
B__inference_model_layer_call_and_return_conditional_losses_9739505?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
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
'__inference_model_layer_call_fn_9738760?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????????<
5?2
(?%
input??????????
p

 
? " ??????????R?
'__inference_model_layer_call_fn_9738996?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~???????????????<
5?2
(?%
input??????????
p 

 
? " ??????????R?
'__inference_model_layer_call_fn_9739618?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
6?3
)?&
inputs??????????
p

 
? " ??????????R?
'__inference_model_layer_call_fn_9739731?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
6?3
)?&
inputs??????????
p 

 
? " ??????????R?
C__inference_relu_0_layer_call_and_return_conditional_losses_9740100h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
(__inference_relu_0_layer_call_fn_9740105[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
C__inference_relu_1_layer_call_and_return_conditional_losses_9740324h7?4
-?*
(?%
inputs?????????R@
? "-?*
#? 
0?????????R@
? ?
(__inference_relu_1_layer_call_fn_9740329[7?4
-?*
(?%
inputs?????????R@
? " ??????????R@?
H__inference_res_block_0_layer_call_and_return_conditional_losses_9736602?cdefghijklmnopqrstG?D
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
H__inference_res_block_0_layer_call_and_return_conditional_losses_9736650?cdefghijklmnopqrstG?D
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
H__inference_res_block_0_layer_call_and_return_conditional_losses_9739947?cdefghijklmnopqrst??<
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
H__inference_res_block_0_layer_call_and_return_conditional_losses_9740013?cdefghijklmnopqrst??<
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
-__inference_res_block_0_layer_call_fn_9736740cdefghijklmnopqrstG?D
=?:
0?-
conv2d_2_input?????????R@
p

 
? " ??????????R@?
-__inference_res_block_0_layer_call_fn_9736829cdefghijklmnopqrstG?D
=?:
0?-
conv2d_2_input?????????R@
p 

 
? " ??????????R@?
-__inference_res_block_0_layer_call_fn_9740054wcdefghijklmnopqrst??<
5?2
(?%
inputs?????????R@
p

 
? " ??????????R@?
-__inference_res_block_0_layer_call_fn_9740095wcdefghijklmnopqrst??<
5?2
(?%
inputs?????????R@
p 

 
? " ??????????R@?
H__inference_res_block_1_layer_call_and_return_conditional_losses_9737451?uvwxyz{|}~???????G?D
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
H__inference_res_block_1_layer_call_and_return_conditional_losses_9737499?uvwxyz{|}~???????G?D
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
H__inference_res_block_1_layer_call_and_return_conditional_losses_9740171?uvwxyz{|}~?????????<
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
H__inference_res_block_1_layer_call_and_return_conditional_losses_9740237?uvwxyz{|}~?????????<
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
-__inference_res_block_1_layer_call_fn_9737589?uvwxyz{|}~???????G?D
=?:
0?-
conv2d_5_input?????????R@
p

 
? " ??????????R@?
-__inference_res_block_1_layer_call_fn_9737678?uvwxyz{|}~???????G?D
=?:
0?-
conv2d_5_input?????????R@
p 

 
? " ??????????R@?
-__inference_res_block_1_layer_call_fn_9740278~uvwxyz{|}~?????????<
5?2
(?%
inputs?????????R@
p

 
? " ??????????R@?
-__inference_res_block_1_layer_call_fn_9740319~uvwxyz{|}~?????????<
5?2
(?%
inputs?????????R@
p 

 
? " ??????????R@?
%__inference_signature_wrapper_9739111?CWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~?????????????@?=
? 
6?3
1
input(?%
input??????????"G?D
B
feature_linear0?-
feature_linear?????????R