ЏЗ:
ѕ§
B
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
Ы
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
ъ
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
epsilonfloat%Ј—8"&
exponential_avg_factorfloat%  А?";
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
delete_old_dirsbool(И
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
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Њ
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
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
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12v2.4.0-49-g85c8b2a817f8ЕИ.
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
Д
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:R*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:R*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:*
dtype0
Р
batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_11/gamma
Й
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes
:*
dtype0
О
batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_11/beta
З
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes
:*
dtype0

conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d/kernel
x
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*'
_output_shapes
:А*
dtype0
o
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes	
:А*
dtype0
Л
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_namebatch_normalization/gamma
Д
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:А*
dtype0
Й
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_namebatch_normalization/beta
В
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:А*
dtype0
Ч
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!batch_normalization/moving_mean
Р
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:А*
dtype0
Я
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#batch_normalization/moving_variance
Ш
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:А*
dtype0
Г
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А@* 
shared_nameconv2d_1/kernel
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*'
_output_shapes
:А@*
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
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
В
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
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
В
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
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
В
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
О
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma
З
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta
Е
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean
У
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance
Ы
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
В
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
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance
Ы
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
В
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
О
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_6/gamma
З
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_6/beta
Е
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_6/moving_mean
У
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_6/moving_variance
Ы
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:@*
dtype0
В
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
О
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_7/gamma
З
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_7/beta
Е
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_7/moving_mean
У
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_7/moving_variance
Ы
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:@*
dtype0
В
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
О
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_8/gamma
З
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_8/beta
Е
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_8/moving_mean
У
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:*
dtype0
Ґ
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_8/moving_variance
Ы
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:*
dtype0
Ь
"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_11/moving_mean
Х
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes
:*
dtype0
§
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_11/moving_variance
Э
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes
:*
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
Т
Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:R*(
shared_nameAdam/conv2d_11/kernel/m
Л
+Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/m*&
_output_shapes
:R*
dtype0
В
Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_11/bias/m
{
)Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/m*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_11/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_11/gamma/m
Ч
7Adam/batch_normalization_11/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_11/gamma/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_11/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_11/beta/m
Х
6Adam/batch_normalization_11/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_11/beta/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:R*(
shared_nameAdam/conv2d_11/kernel/v
Л
+Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/v*&
_output_shapes
:R*
dtype0
В
Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_11/bias/v
{
)Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/v*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_11/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_11/gamma/v
Ч
7Adam/batch_normalization_11/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_11/gamma/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_11/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_11/beta/v
Х
6Adam/batch_normalization_11/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_11/beta/v*
_output_shapes
:*
dtype0

NoOpNoOp
®ї
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*вЇ
value„ЇB”Ї BЋЇ
≠
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
layer_with_weights-4
layer-11
layer-12
layer-13
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
И
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
trainable_variables
regularization_losses
	variables
	keras_api
÷
layer_with_weights-0
layer-0
 layer_with_weights-1
 layer-1
!layer-2
"layer_with_weights-2
"layer-3
#layer_with_weights-3
#layer-4
$layer-5
%layer_with_weights-4
%layer-6
&layer_with_weights-5
&layer-7
'trainable_variables
(regularization_losses
)	variables
*	keras_api

+	keras_api
R
,trainable_variables
-regularization_losses
.	variables
/	keras_api
÷
0layer_with_weights-0
0layer-0
1layer_with_weights-1
1layer-1
2layer-2
3layer_with_weights-2
3layer-3
4layer_with_weights-3
4layer-4
5layer-5
6layer_with_weights-4
6layer-6
7layer_with_weights-5
7layer-7
8trainable_variables
9regularization_losses
:	variables
;	keras_api

<	keras_api
R
=trainable_variables
>regularization_losses
?	variables
@	keras_api
≠
Alayer_with_weights-0
Alayer-0
Blayer_with_weights-1
Blayer-1
Clayer-2
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api

H	keras_api
R
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
≠
Mlayer_with_weights-0
Mlayer-0
Nlayer_with_weights-1
Nlayer-1
Olayer-2
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
R
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
R
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
}
\iter

]beta_1

^beta_2
	_decay`mдamеbmжcmз`vиavйbvкcvл

`0
a1
b2
c3
 
т
d0
e1
f2
g3
h4
i5
j6
k7
l8
m9
n10
o11
p12
q13
r14
s15
t16
u17
v18
w19
x20
y21
z22
{23
|24
}25
~26
27
А28
Б29
В30
Г31
Д32
Е33
Ж34
З35
И36
Й37
К38
Л39
М40
Н41
О42
П43
Р44
С45
Т46
У47
Ф48
Х49
Ц50
Ч51
Ш52
Щ53
`54
a55
b56
c57
Ъ58
Ы59
≤
trainable_variables
Ьnon_trainable_variables
Эmetrics
Юlayers
regularization_losses
 Яlayer_regularization_losses
†layer_metrics
	variables
 
l

dkernel
ebias
°trainable_variables
Ґregularization_losses
£	variables
§	keras_api
Ь
	•axis
	fgamma
gbeta
hmoving_mean
imoving_variance
¶trainable_variables
Іregularization_losses
®	variables
©	keras_api
V
™trainable_variables
Ђregularization_losses
ђ	variables
≠	keras_api
l

jkernel
kbias
Ѓtrainable_variables
ѓregularization_losses
∞	variables
±	keras_api
Ь
	≤axis
	lgamma
mbeta
nmoving_mean
omoving_variance
≥trainable_variables
іregularization_losses
µ	variables
ґ	keras_api
V
Јtrainable_variables
Єregularization_losses
є	variables
Ї	keras_api
 
 
V
d0
e1
f2
g3
h4
i5
j6
k7
l8
m9
n10
o11
≤
trainable_variables
їnon_trainable_variables
Љmetrics
љlayers
regularization_losses
 Њlayer_regularization_losses
њlayer_metrics
	variables
l

pkernel
qbias
јtrainable_variables
Ѕregularization_losses
¬	variables
√	keras_api
Ь
	ƒaxis
	rgamma
sbeta
tmoving_mean
umoving_variance
≈trainable_variables
∆regularization_losses
«	variables
»	keras_api
V
…trainable_variables
 regularization_losses
Ћ	variables
ћ	keras_api
l

vkernel
wbias
Ќtrainable_variables
ќregularization_losses
ѕ	variables
–	keras_api
Ь
	—axis
	xgamma
ybeta
zmoving_mean
{moving_variance
“trainable_variables
”regularization_losses
‘	variables
’	keras_api
V
÷trainable_variables
„regularization_losses
Ў	variables
ў	keras_api
l

|kernel
}bias
Џtrainable_variables
џregularization_losses
№	variables
Ё	keras_api
Ю
	ёaxis
	~gamma
beta
Аmoving_mean
Бmoving_variance
яtrainable_variables
аregularization_losses
б	variables
в	keras_api
 
 
И
p0
q1
r2
s3
t4
u5
v6
w7
x8
y9
z10
{11
|12
}13
~14
15
А16
Б17
≤
'trainable_variables
гnon_trainable_variables
дmetrics
еlayers
(regularization_losses
 жlayer_regularization_losses
зlayer_metrics
)	variables
 
 
 
 
≤
,trainable_variables
иmetrics
йnon_trainable_variables
кlayers
-regularization_losses
 лlayer_regularization_losses
мlayer_metrics
.	variables
n
Вkernel
	Гbias
нtrainable_variables
оregularization_losses
п	variables
р	keras_api
†
	сaxis

Дgamma
	Еbeta
Жmoving_mean
Зmoving_variance
тtrainable_variables
уregularization_losses
ф	variables
х	keras_api
V
цtrainable_variables
чregularization_losses
ш	variables
щ	keras_api
n
Иkernel
	Йbias
ъtrainable_variables
ыregularization_losses
ь	variables
э	keras_api
†
	юaxis

Кgamma
	Лbeta
Мmoving_mean
Нmoving_variance
€trainable_variables
Аregularization_losses
Б	variables
В	keras_api
V
Гtrainable_variables
Дregularization_losses
Е	variables
Ж	keras_api
n
Оkernel
	Пbias
Зtrainable_variables
Иregularization_losses
Й	variables
К	keras_api
†
	Лaxis

Рgamma
	Сbeta
Тmoving_mean
Уmoving_variance
Мtrainable_variables
Нregularization_losses
О	variables
П	keras_api
 
 
Ш
В0
Г1
Д2
Е3
Ж4
З5
И6
Й7
К8
Л9
М10
Н11
О12
П13
Р14
С15
Т16
У17
≤
8trainable_variables
Рnon_trainable_variables
Сmetrics
Тlayers
9regularization_losses
 Уlayer_regularization_losses
Фlayer_metrics
:	variables
 
 
 
 
≤
=trainable_variables
Хmetrics
Цnon_trainable_variables
Чlayers
>regularization_losses
 Шlayer_regularization_losses
Щlayer_metrics
?	variables
n
Фkernel
	Хbias
Ъtrainable_variables
Ыregularization_losses
Ь	variables
Э	keras_api
†
	Юaxis

Цgamma
	Чbeta
Шmoving_mean
Щmoving_variance
Яtrainable_variables
†regularization_losses
°	variables
Ґ	keras_api
V
£trainable_variables
§regularization_losses
•	variables
¶	keras_api
 
 
0
Ф0
Х1
Ц2
Ч3
Ш4
Щ5
≤
Dtrainable_variables
Іnon_trainable_variables
®metrics
©layers
Eregularization_losses
 ™layer_regularization_losses
Ђlayer_metrics
F	variables
 
 
 
 
≤
Itrainable_variables
ђmetrics
≠non_trainable_variables
Ѓlayers
Jregularization_losses
 ѓlayer_regularization_losses
∞layer_metrics
K	variables
l

`kernel
abias
±trainable_variables
≤regularization_losses
≥	variables
і	keras_api
Ю
	µaxis
	bgamma
cbeta
Ъmoving_mean
Ыmoving_variance
ґtrainable_variables
Јregularization_losses
Є	variables
є	keras_api
V
Їtrainable_variables
їregularization_losses
Љ	variables
љ	keras_api

`0
a1
b2
c3
 
,
`0
a1
b2
c3
Ъ4
Ы5
≤
Ptrainable_variables
Њnon_trainable_variables
њmetrics
јlayers
Qregularization_losses
 Ѕlayer_regularization_losses
¬layer_metrics
R	variables
 
 
 
≤
Ttrainable_variables
√metrics
ƒnon_trainable_variables
≈layers
Uregularization_losses
 ∆layer_regularization_losses
«layer_metrics
V	variables
 
 
 
≤
Xtrainable_variables
»metrics
…non_trainable_variables
 layers
Yregularization_losses
 Ћlayer_regularization_losses
ћlayer_metrics
Z	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_11/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_11/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_11/gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_11/beta0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
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
_]
VARIABLE_VALUE"batch_normalization_11/moving_mean'variables/58/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_11/moving_variance'variables/59/.ATTRIBUTES/VARIABLE_VALUE
“
d0
e1
f2
g3
h4
i5
j6
k7
l8
m9
n10
o11
p12
q13
r14
s15
t16
u17
v18
w19
x20
y21
z22
{23
|24
}25
~26
27
А28
Б29
В30
Г31
Д32
Е33
Ж34
З35
И36
Й37
К38
Л39
М40
Н41
О42
П43
Р44
С45
Т46
У47
Ф48
Х49
Ц50
Ч51
Ш52
Щ53
Ъ54
Ы55

Ќ0
ќ1
f
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
11
12
13
 
 
 
 

d0
e1
µ
°trainable_variables
ѕmetrics
–non_trainable_variables
—layers
Ґregularization_losses
 “layer_regularization_losses
”layer_metrics
£	variables
 
 
 

f0
g1
h2
i3
µ
¶trainable_variables
‘metrics
’non_trainable_variables
÷layers
Іregularization_losses
 „layer_regularization_losses
Ўlayer_metrics
®	variables
 
 
 
µ
™trainable_variables
ўmetrics
Џnon_trainable_variables
џlayers
Ђregularization_losses
 №layer_regularization_losses
Ёlayer_metrics
ђ	variables
 
 

j0
k1
µ
Ѓtrainable_variables
ёmetrics
яnon_trainable_variables
аlayers
ѓregularization_losses
 бlayer_regularization_losses
вlayer_metrics
∞	variables
 
 
 

l0
m1
n2
o3
µ
≥trainable_variables
гmetrics
дnon_trainable_variables
еlayers
іregularization_losses
 жlayer_regularization_losses
зlayer_metrics
µ	variables
 
 
 
µ
Јtrainable_variables
иmetrics
йnon_trainable_variables
кlayers
Єregularization_losses
 лlayer_regularization_losses
мlayer_metrics
є	variables
V
d0
e1
f2
g3
h4
i5
j6
k7
l8
m9
n10
o11
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
 

p0
q1
µ
јtrainable_variables
нmetrics
оnon_trainable_variables
пlayers
Ѕregularization_losses
 рlayer_regularization_losses
сlayer_metrics
¬	variables
 
 
 

r0
s1
t2
u3
µ
≈trainable_variables
тmetrics
уnon_trainable_variables
фlayers
∆regularization_losses
 хlayer_regularization_losses
цlayer_metrics
«	variables
 
 
 
µ
…trainable_variables
чmetrics
шnon_trainable_variables
щlayers
 regularization_losses
 ъlayer_regularization_losses
ыlayer_metrics
Ћ	variables
 
 

v0
w1
µ
Ќtrainable_variables
ьmetrics
эnon_trainable_variables
юlayers
ќregularization_losses
 €layer_regularization_losses
Аlayer_metrics
ѕ	variables
 
 
 

x0
y1
z2
{3
µ
“trainable_variables
Бmetrics
Вnon_trainable_variables
Гlayers
”regularization_losses
 Дlayer_regularization_losses
Еlayer_metrics
‘	variables
 
 
 
µ
÷trainable_variables
Жmetrics
Зnon_trainable_variables
Иlayers
„regularization_losses
 Йlayer_regularization_losses
Кlayer_metrics
Ў	variables
 
 

|0
}1
µ
Џtrainable_variables
Лmetrics
Мnon_trainable_variables
Нlayers
џregularization_losses
 Оlayer_regularization_losses
Пlayer_metrics
№	variables
 
 
 

~0
1
А2
Б3
µ
яtrainable_variables
Рmetrics
Сnon_trainable_variables
Тlayers
аregularization_losses
 Уlayer_regularization_losses
Фlayer_metrics
б	variables
И
p0
q1
r2
s3
t4
u5
v6
w7
x8
y9
z10
{11
|12
}13
~14
15
А16
Б17
 
8
0
 1
!2
"3
#4
$5
%6
&7
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
В0
Г1
µ
нtrainable_variables
Хmetrics
Цnon_trainable_variables
Чlayers
оregularization_losses
 Шlayer_regularization_losses
Щlayer_metrics
п	variables
 
 
 
 
Д0
Е1
Ж2
З3
µ
тtrainable_variables
Ъmetrics
Ыnon_trainable_variables
Ьlayers
уregularization_losses
 Эlayer_regularization_losses
Юlayer_metrics
ф	variables
 
 
 
µ
цtrainable_variables
Яmetrics
†non_trainable_variables
°layers
чregularization_losses
 Ґlayer_regularization_losses
£layer_metrics
ш	variables
 
 

И0
Й1
µ
ъtrainable_variables
§metrics
•non_trainable_variables
¶layers
ыregularization_losses
 Іlayer_regularization_losses
®layer_metrics
ь	variables
 
 
 
 
К0
Л1
М2
Н3
µ
€trainable_variables
©metrics
™non_trainable_variables
Ђlayers
Аregularization_losses
 ђlayer_regularization_losses
≠layer_metrics
Б	variables
 
 
 
µ
Гtrainable_variables
Ѓmetrics
ѓnon_trainable_variables
∞layers
Дregularization_losses
 ±layer_regularization_losses
≤layer_metrics
Е	variables
 
 

О0
П1
µ
Зtrainable_variables
≥metrics
іnon_trainable_variables
µlayers
Иregularization_losses
 ґlayer_regularization_losses
Јlayer_metrics
Й	variables
 
 
 
 
Р0
С1
Т2
У3
µ
Мtrainable_variables
Єmetrics
єnon_trainable_variables
Їlayers
Нregularization_losses
 їlayer_regularization_losses
Љlayer_metrics
О	variables
Ш
В0
Г1
Д2
Е3
Ж4
З5
И6
Й7
К8
Л9
М10
Н11
О12
П13
Р14
С15
Т16
У17
 
8
00
11
22
33
44
55
66
77
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
Ф0
Х1
µ
Ъtrainable_variables
љmetrics
Њnon_trainable_variables
њlayers
Ыregularization_losses
 јlayer_regularization_losses
Ѕlayer_metrics
Ь	variables
 
 
 
 
Ц0
Ч1
Ш2
Щ3
µ
Яtrainable_variables
¬metrics
√non_trainable_variables
ƒlayers
†regularization_losses
 ≈layer_regularization_losses
∆layer_metrics
°	variables
 
 
 
µ
£trainable_variables
«metrics
»non_trainable_variables
…layers
§regularization_losses
  layer_regularization_losses
Ћlayer_metrics
•	variables
0
Ф0
Х1
Ц2
Ч3
Ш4
Щ5
 

A0
B1
C2
 
 
 
 
 
 
 

`0
a1
 

`0
a1
µ
±trainable_variables
ћmetrics
Ќnon_trainable_variables
ќlayers
≤regularization_losses
 ѕlayer_regularization_losses
–layer_metrics
≥	variables
 

b0
c1
 

b0
c1
Ъ2
Ы3
µ
ґtrainable_variables
—metrics
“non_trainable_variables
”layers
Јregularization_losses
 ‘layer_regularization_losses
’layer_metrics
Є	variables
 
 
 
µ
Їtrainable_variables
÷metrics
„non_trainable_variables
Ўlayers
їregularization_losses
 ўlayer_regularization_losses
Џlayer_metrics
Љ	variables

Ъ0
Ы1
 

M0
N1
O2
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
8

џtotal

№count
Ё	variables
ё	keras_api
I

яtotal

аcount
б
_fn_kwargs
в	variables
г	keras_api
 

d0
e1
 
 
 
 

f0
g1
h2
i3
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
j0
k1
 
 
 
 

l0
m1
n2
o3
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
p0
q1
 
 
 
 

r0
s1
t2
u3
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
v0
w1
 
 
 
 

x0
y1
z2
{3
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
|0
}1
 
 
 
 

~0
1
А2
Б3
 
 
 
 

В0
Г1
 
 
 
 
 
Д0
Е1
Ж2
З3
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
И0
Й1
 
 
 
 
 
К0
Л1
М2
Н3
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
О0
П1
 
 
 
 
 
Р0
С1
Т2
У3
 
 
 
 

Ф0
Х1
 
 
 
 
 
Ц0
Ч1
Ш2
Щ3
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

Ъ0
Ы1
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

џ0
№1

Ё	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

я0
а1

в	variables
yw
VARIABLE_VALUEAdam/conv2d_11/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_11/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE#Adam/batch_normalization_11/gamma/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/batch_normalization_11/beta/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_11/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_11/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE#Adam/batch_normalization_11/gamma/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/batch_normalization_11/beta/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_inputPlaceholder*0
_output_shapes
:€€€€€€€€€ь*
dtype0*%
shape:€€€€€€€€€ь
З
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_variance*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*2
config_proto" 

CPU

GPU2 *0J 8В */
f*R(
&__inference_signature_wrapper_34129112
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
щ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_11/kernel/m/Read/ReadVariableOp)Adam/conv2d_11/bias/m/Read/ReadVariableOp7Adam/batch_normalization_11/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_11/beta/m/Read/ReadVariableOp+Adam/conv2d_11/kernel/v/Read/ReadVariableOp)Adam/conv2d_11/bias/v/Read/ReadVariableOp7Adam/batch_normalization_11/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_11/beta/v/Read/ReadVariableOpConst*Y
TinR
P2N	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В **
f%R#
!__inference__traced_save_34132398
Д
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayconv2d_11/kernelconv2d_11/biasbatch_normalization_11/gammabatch_normalization_11/betaconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variance"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancetotalcounttotal_1count_1Adam/conv2d_11/kernel/mAdam/conv2d_11/bias/m#Adam/batch_normalization_11/gamma/m"Adam/batch_normalization_11/beta/mAdam/conv2d_11/kernel/vAdam/conv2d_11/bias/v#Adam/batch_normalization_11/gamma/v"Adam/batch_normalization_11/beta/v*X
TinQ
O2M*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *-
f(R&
$__inference__traced_restore_34132636Шй*
П"
€
H__inference_input_conv_layer_call_and_return_conditional_losses_34125375
conv2d_input
conv2d_34125344
conv2d_34125346 
batch_normalization_34125349 
batch_normalization_34125351 
batch_normalization_34125353 
batch_normalization_34125355
conv2d_1_34125359
conv2d_1_34125361"
batch_normalization_1_34125364"
batch_normalization_1_34125366"
batch_normalization_1_34125368"
batch_normalization_1_34125370
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCall•
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_34125344conv2d_34125346*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_341251302 
conv2d/StatefulPartitionedCallЅ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_34125349batch_normalization_34125351batch_normalization_34125353batch_normalization_34125355*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_341251812-
+batch_normalization/StatefulPartitionedCallЩ
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_341252222
activation/PartitionedCallƒ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_34125359conv2d_1_34125361*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_341252402"
 conv2d_1/StatefulPartitionedCallѕ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_34125364batch_normalization_1_34125366batch_normalization_1_34125368batch_normalization_1_34125370*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_341252912/
-batch_normalization_1/StatefulPartitionedCallЯ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_341253322
activation_1/PartitionedCall£
IdentityIdentity%activation_1/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€ь::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:^ Z
0
_output_shapes
:€€€€€€€€€ь
&
_user_specified_nameconv2d_input
Ж
А
+__inference_conv2d_5_layer_call_fn_34131400

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_341266652
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ќ
ф
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34125005

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3н
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34131184

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34127289

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∞К
љ"
!__inference__traced_save_34132398
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableop,
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
@savev2_batch_normalization_8_moving_variance_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_11_kernel_m_read_readvariableop4
0savev2_adam_conv2d_11_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_11_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_11_beta_m_read_readvariableop6
2savev2_adam_conv2d_11_kernel_v_read_readvariableop4
0savev2_adam_conv2d_11_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_11_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_11_beta_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename–
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*в
valueЎB’MB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names•
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*ѓ
value•BҐMB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices•!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_11_kernel_m_read_readvariableop0savev2_adam_conv2d_11_bias_m_read_readvariableop>savev2_adam_batch_normalization_11_gamma_m_read_readvariableop=savev2_adam_batch_normalization_11_beta_m_read_readvariableop2savev2_adam_conv2d_11_kernel_v_read_readvariableop0savev2_adam_conv2d_11_bias_v_read_readvariableop>savev2_adam_batch_normalization_11_gamma_v_read_readvariableop=savev2_adam_batch_normalization_11_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *[
dtypesQ
O2M	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*Ў
_input_shapes∆
√: : : : : :R::::А:А:А:А:А:А:А@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@:::::::: : : : :R::::R:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:R: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::-	)
'
_output_shapes
:А:!


_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:-)
'
_output_shapes
:А@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:  

_output_shapes
:@:,!(
&
_output_shapes
:@@: "

_output_shapes
:@: #

_output_shapes
:@: $

_output_shapes
:@: %

_output_shapes
:@: &

_output_shapes
:@:,'(
&
_output_shapes
:@@: (

_output_shapes
:@: )

_output_shapes
:@: *

_output_shapes
:@: +

_output_shapes
:@: ,

_output_shapes
:@:,-(
&
_output_shapes
:@@: .

_output_shapes
:@: /

_output_shapes
:@: 0

_output_shapes
:@: 1

_output_shapes
:@: 2

_output_shapes
:@:,3(
&
_output_shapes
:@@: 4

_output_shapes
:@: 5

_output_shapes
:@: 6

_output_shapes
:@: 7

_output_shapes
:@: 8

_output_shapes
:@:,9(
&
_output_shapes
:@: :

_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
:: >

_output_shapes
:: ?

_output_shapes
:: @

_output_shapes
::A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :,E(
&
_output_shapes
:R: F

_output_shapes
:: G

_output_shapes
:: H

_output_shapes
::,I(
&
_output_shapes
:R: J

_output_shapes
:: K

_output_shapes
:: L

_output_shapes
::M

_output_shapes
: 
Ў
f
J__inference_activation_4_layer_call_and_return_conditional_losses_34126757

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
о	
а
G__inference_conv2d_11_layer_call_and_return_conditional_losses_34131993

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
®
Ђ
8__inference_batch_normalization_5_layer_call_fn_34131449

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_341264092
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
®
Ђ
8__inference_batch_normalization_6_layer_call_fn_34131602

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_341265092
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Й
ф
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34125181

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1–
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:€€€€€€€€€®А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3№
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:€€€€€€€€€®А2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:€€€€€€€€€®А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:€€€€€€€€€®А
 
_user_specified_nameinputs
м	
я
F__inference_conv2d_4_layer_call_and_return_conditional_losses_34131248

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Д/
Ы
I__inference_res_block_1_layer_call_and_return_conditional_losses_34127072

inputs
conv2d_5_34127027
conv2d_5_34127029"
batch_normalization_5_34127032"
batch_normalization_5_34127034"
batch_normalization_5_34127036"
batch_normalization_5_34127038
conv2d_6_34127042
conv2d_6_34127044"
batch_normalization_6_34127047"
batch_normalization_6_34127049"
batch_normalization_6_34127051"
batch_normalization_6_34127053
conv2d_7_34127057
conv2d_7_34127059"
batch_normalization_7_34127062"
batch_normalization_7_34127064"
batch_normalization_7_34127066"
batch_normalization_7_34127068
identityИҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐ-batch_normalization_7/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallІ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_34127027conv2d_5_34127029*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_341266652"
 conv2d_5/StatefulPartitionedCallѕ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_34127032batch_normalization_5_34127034batch_normalization_5_34127036batch_normalization_5_34127038*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_341266982/
-batch_normalization_5/StatefulPartitionedCallЯ
activation_4/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_4_layer_call_and_return_conditional_losses_341267572
activation_4/PartitionedCall∆
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_34127042conv2d_6_34127044*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_341267752"
 conv2d_6/StatefulPartitionedCallѕ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_34127047batch_normalization_6_34127049batch_normalization_6_34127051batch_normalization_6_34127053*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_341268082/
-batch_normalization_6/StatefulPartitionedCallЯ
activation_5/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_5_layer_call_and_return_conditional_losses_341268672
activation_5/PartitionedCall∆
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_34127057conv2d_7_34127059*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_7_layer_call_and_return_conditional_losses_341268852"
 conv2d_7/StatefulPartitionedCallѕ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_34127062batch_normalization_7_34127064batch_normalization_7_34127066batch_normalization_7_34127068*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_341269182/
-batch_normalization_7/StatefulPartitionedCallЛ
IdentityIdentity6batch_normalization_7/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34131498

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34131140

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ж
А
+__inference_conv2d_2_layer_call_fn_34130951

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_341258162
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ў
f
J__inference_activation_1_layer_call_and_return_conditional_losses_34130927

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
а
Ђ
8__inference_batch_normalization_8_layer_call_fn_34131898

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_341273472
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
Щ
F
*__inference_softmax_layer_call_fn_34130626

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_341283052
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
•
ґ
E__inference_feature_layer_call_and_return_conditional_losses_34130439

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identityИҐ5batch_normalization_8/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_8/ReadVariableOpҐ&batch_normalization_8/ReadVariableOp_1Ґconv2d_8/BiasAdd/ReadVariableOpҐconv2d_8/Conv2D/ReadVariableOp∞
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_8/Conv2D/ReadVariableOpњ
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R*
paddingVALID*
strides
2
conv2d_8/Conv2DІ
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOpђ
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R2
conv2d_8/BiasAddґ
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOpЉ
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1й
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3Ф
activation_6/ReluRelu*batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R2
activation_6/ReluА
IdentityIdentityactivation_6/Relu:activations:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1 ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R@::::::2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
м	
я
F__inference_conv2d_6_layer_call_and_return_conditional_losses_34131544

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
м	
я
F__inference_conv2d_2_layer_call_and_return_conditional_losses_34125816

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ы
Ц
*__inference_model_1_layer_call_fn_34129816

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

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58
identityИҐStatefulPartitionedCallЭ	
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
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_341288562
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*°
_input_shapesП
М:€€€€€€€€€ь::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ь
 
_user_specified_nameinputs
Ъ
ъ
.__inference_res_block_0_layer_call_fn_34130180

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
identityИҐStatefulPartitionedCallЁ
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
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_341263122
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
т
К
I__inference_output_conv_layer_call_and_return_conditional_losses_34127754
conv2d_11_input
conv2d_11_34127738
conv2d_11_34127740#
batch_normalization_11_34127743#
batch_normalization_11_34127745#
batch_normalization_11_34127747#
batch_normalization_11_34127749
identityИҐ.batch_normalization_11/StatefulPartitionedCallҐ!conv2d_11/StatefulPartitionedCallµ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallconv2d_11_inputconv2d_11_34127738conv2d_11_34127740*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_341276252#
!conv2d_11/StatefulPartitionedCall„
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_11_34127743batch_normalization_11_34127745batch_normalization_11_34127747batch_normalization_11_34127749*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *]
fXRV
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3412767820
.batch_normalization_11/StatefulPartitionedCallП
reshape_1/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_341277262
reshape_1/PartitionedCallЋ
IdentityIdentity"reshape_1/PartitionedCall:output:0/^batch_normalization_11/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R::::::2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall:` \
/
_output_shapes
:€€€€€€€€€R
)
_user_specified_nameconv2d_11_input
®
©
6__inference_batch_normalization_layer_call_fn_34130707

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_341250052
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ш
Х
*__inference_model_1_layer_call_fn_34128979	
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

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58
identityИҐStatefulPartitionedCallЬ	
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
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_341288562
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*°
_input_shapesП
М:€€€€€€€€€ь::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:€€€€€€€€€ь

_user_specified_nameinput
®
Ђ
8__inference_batch_normalization_4_layer_call_fn_34131319

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_341257912
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
√
K
/__inference_activation_3_layer_call_fn_34131238

inputs
identity’
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_3_layer_call_and_return_conditional_losses_341260182
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34125791

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
а
Ђ
8__inference_batch_normalization_6_layer_call_fn_34131664

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_341268082
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
’
Б
I__inference_output_conv_layer_call_and_return_conditional_losses_34127776

inputs
conv2d_11_34127760
conv2d_11_34127762#
batch_normalization_11_34127765#
batch_normalization_11_34127767#
batch_normalization_11_34127769#
batch_normalization_11_34127771
identityИҐ.batch_normalization_11/StatefulPartitionedCallҐ!conv2d_11/StatefulPartitionedCallђ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_34127760conv2d_11_34127762*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_341276252#
!conv2d_11/StatefulPartitionedCall’
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_11_34127765batch_normalization_11_34127767batch_normalization_11_34127769batch_normalization_11_34127771*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *]
fXRV
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3412766020
.batch_normalization_11/StatefulPartitionedCallП
reshape_1/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_341277262
reshape_1/PartitionedCallЋ
IdentityIdentity"reshape_1/PartitionedCall:output:0/^batch_normalization_11/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R::::::2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
м	
я
F__inference_conv2d_4_layer_call_and_return_conditional_losses_34126036

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34131122

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34125273

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
”
х
E__inference_feature_layer_call_and_return_conditional_losses_34127456

inputs
conv2d_8_34127440
conv2d_8_34127442"
batch_normalization_8_34127445"
batch_normalization_8_34127447"
batch_normalization_8_34127449"
batch_normalization_8_34127451
identityИҐ-batch_normalization_8/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallІ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_34127440conv2d_8_34127442*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_8_layer_call_and_return_conditional_losses_341273142"
 conv2d_8/StatefulPartitionedCallѕ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_34127445batch_normalization_8_34127447batch_normalization_8_34127449batch_normalization_8_34127451*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_341273472/
-batch_normalization_8/StatefulPartitionedCallЯ
activation_6/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_6_layer_call_and_return_conditional_losses_341274062
activation_6/PartitionedCall‘
IdentityIdentity%activation_6/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R@::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Гґ
ц*
$__inference__traced_restore_34132636
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay'
#assignvariableop_4_conv2d_11_kernel%
!assignvariableop_5_conv2d_11_bias3
/assignvariableop_6_batch_normalization_11_gamma2
.assignvariableop_7_batch_normalization_11_beta$
 assignvariableop_8_conv2d_kernel"
assignvariableop_9_conv2d_bias1
-assignvariableop_10_batch_normalization_gamma0
,assignvariableop_11_batch_normalization_beta7
3assignvariableop_12_batch_normalization_moving_mean;
7assignvariableop_13_batch_normalization_moving_variance'
#assignvariableop_14_conv2d_1_kernel%
!assignvariableop_15_conv2d_1_bias3
/assignvariableop_16_batch_normalization_1_gamma2
.assignvariableop_17_batch_normalization_1_beta9
5assignvariableop_18_batch_normalization_1_moving_mean=
9assignvariableop_19_batch_normalization_1_moving_variance'
#assignvariableop_20_conv2d_2_kernel%
!assignvariableop_21_conv2d_2_bias3
/assignvariableop_22_batch_normalization_2_gamma2
.assignvariableop_23_batch_normalization_2_beta9
5assignvariableop_24_batch_normalization_2_moving_mean=
9assignvariableop_25_batch_normalization_2_moving_variance'
#assignvariableop_26_conv2d_3_kernel%
!assignvariableop_27_conv2d_3_bias3
/assignvariableop_28_batch_normalization_3_gamma2
.assignvariableop_29_batch_normalization_3_beta9
5assignvariableop_30_batch_normalization_3_moving_mean=
9assignvariableop_31_batch_normalization_3_moving_variance'
#assignvariableop_32_conv2d_4_kernel%
!assignvariableop_33_conv2d_4_bias3
/assignvariableop_34_batch_normalization_4_gamma2
.assignvariableop_35_batch_normalization_4_beta9
5assignvariableop_36_batch_normalization_4_moving_mean=
9assignvariableop_37_batch_normalization_4_moving_variance'
#assignvariableop_38_conv2d_5_kernel%
!assignvariableop_39_conv2d_5_bias3
/assignvariableop_40_batch_normalization_5_gamma2
.assignvariableop_41_batch_normalization_5_beta9
5assignvariableop_42_batch_normalization_5_moving_mean=
9assignvariableop_43_batch_normalization_5_moving_variance'
#assignvariableop_44_conv2d_6_kernel%
!assignvariableop_45_conv2d_6_bias3
/assignvariableop_46_batch_normalization_6_gamma2
.assignvariableop_47_batch_normalization_6_beta9
5assignvariableop_48_batch_normalization_6_moving_mean=
9assignvariableop_49_batch_normalization_6_moving_variance'
#assignvariableop_50_conv2d_7_kernel%
!assignvariableop_51_conv2d_7_bias3
/assignvariableop_52_batch_normalization_7_gamma2
.assignvariableop_53_batch_normalization_7_beta9
5assignvariableop_54_batch_normalization_7_moving_mean=
9assignvariableop_55_batch_normalization_7_moving_variance'
#assignvariableop_56_conv2d_8_kernel%
!assignvariableop_57_conv2d_8_bias3
/assignvariableop_58_batch_normalization_8_gamma2
.assignvariableop_59_batch_normalization_8_beta9
5assignvariableop_60_batch_normalization_8_moving_mean=
9assignvariableop_61_batch_normalization_8_moving_variance:
6assignvariableop_62_batch_normalization_11_moving_mean>
:assignvariableop_63_batch_normalization_11_moving_variance
assignvariableop_64_total
assignvariableop_65_count
assignvariableop_66_total_1
assignvariableop_67_count_1/
+assignvariableop_68_adam_conv2d_11_kernel_m-
)assignvariableop_69_adam_conv2d_11_bias_m;
7assignvariableop_70_adam_batch_normalization_11_gamma_m:
6assignvariableop_71_adam_batch_normalization_11_beta_m/
+assignvariableop_72_adam_conv2d_11_kernel_v-
)assignvariableop_73_adam_conv2d_11_bias_v;
7assignvariableop_74_adam_batch_normalization_11_gamma_v:
6assignvariableop_75_adam_batch_normalization_11_beta_v
identity_77ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_8ҐAssignVariableOp_9÷
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*в
valueЎB’MB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/58/.ATTRIBUTES/VARIABLE_VALUEB'variables/59/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЂ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*ѓ
value•BҐMB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesѓ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapesЈ
і:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*[
dtypesQ
O2M	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

IdentityЩ
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2£
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ґ
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4®
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_11_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¶
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_11_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6і
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_11_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7≥
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_11_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8•
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2d_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9£
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv2d_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10µ
AssignVariableOp_10AssignVariableOp-assignvariableop_10_batch_normalization_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11і
AssignVariableOp_11AssignVariableOp,assignvariableop_11_batch_normalization_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ї
AssignVariableOp_12AssignVariableOp3assignvariableop_12_batch_normalization_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13њ
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ђ
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ј
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_1_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ґ
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_1_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18љ
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_1_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ѕ
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_1_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ђ
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv2d_2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21©
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv2d_2_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ј
AssignVariableOp_22AssignVariableOp/assignvariableop_22_batch_normalization_2_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23ґ
AssignVariableOp_23AssignVariableOp.assignvariableop_23_batch_normalization_2_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24љ
AssignVariableOp_24AssignVariableOp5assignvariableop_24_batch_normalization_2_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ѕ
AssignVariableOp_25AssignVariableOp9assignvariableop_25_batch_normalization_2_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ђ
AssignVariableOp_26AssignVariableOp#assignvariableop_26_conv2d_3_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27©
AssignVariableOp_27AssignVariableOp!assignvariableop_27_conv2d_3_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ј
AssignVariableOp_28AssignVariableOp/assignvariableop_28_batch_normalization_3_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ґ
AssignVariableOp_29AssignVariableOp.assignvariableop_29_batch_normalization_3_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30љ
AssignVariableOp_30AssignVariableOp5assignvariableop_30_batch_normalization_3_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ѕ
AssignVariableOp_31AssignVariableOp9assignvariableop_31_batch_normalization_3_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ђ
AssignVariableOp_32AssignVariableOp#assignvariableop_32_conv2d_4_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33©
AssignVariableOp_33AssignVariableOp!assignvariableop_33_conv2d_4_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ј
AssignVariableOp_34AssignVariableOp/assignvariableop_34_batch_normalization_4_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35ґ
AssignVariableOp_35AssignVariableOp.assignvariableop_35_batch_normalization_4_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36љ
AssignVariableOp_36AssignVariableOp5assignvariableop_36_batch_normalization_4_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ѕ
AssignVariableOp_37AssignVariableOp9assignvariableop_37_batch_normalization_4_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ђ
AssignVariableOp_38AssignVariableOp#assignvariableop_38_conv2d_5_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39©
AssignVariableOp_39AssignVariableOp!assignvariableop_39_conv2d_5_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Ј
AssignVariableOp_40AssignVariableOp/assignvariableop_40_batch_normalization_5_gammaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41ґ
AssignVariableOp_41AssignVariableOp.assignvariableop_41_batch_normalization_5_betaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42љ
AssignVariableOp_42AssignVariableOp5assignvariableop_42_batch_normalization_5_moving_meanIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ѕ
AssignVariableOp_43AssignVariableOp9assignvariableop_43_batch_normalization_5_moving_varianceIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Ђ
AssignVariableOp_44AssignVariableOp#assignvariableop_44_conv2d_6_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45©
AssignVariableOp_45AssignVariableOp!assignvariableop_45_conv2d_6_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Ј
AssignVariableOp_46AssignVariableOp/assignvariableop_46_batch_normalization_6_gammaIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47ґ
AssignVariableOp_47AssignVariableOp.assignvariableop_47_batch_normalization_6_betaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48љ
AssignVariableOp_48AssignVariableOp5assignvariableop_48_batch_normalization_6_moving_meanIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ѕ
AssignVariableOp_49AssignVariableOp9assignvariableop_49_batch_normalization_6_moving_varianceIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Ђ
AssignVariableOp_50AssignVariableOp#assignvariableop_50_conv2d_7_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51©
AssignVariableOp_51AssignVariableOp!assignvariableop_51_conv2d_7_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Ј
AssignVariableOp_52AssignVariableOp/assignvariableop_52_batch_normalization_7_gammaIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53ґ
AssignVariableOp_53AssignVariableOp.assignvariableop_53_batch_normalization_7_betaIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54љ
AssignVariableOp_54AssignVariableOp5assignvariableop_54_batch_normalization_7_moving_meanIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Ѕ
AssignVariableOp_55AssignVariableOp9assignvariableop_55_batch_normalization_7_moving_varianceIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Ђ
AssignVariableOp_56AssignVariableOp#assignvariableop_56_conv2d_8_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57©
AssignVariableOp_57AssignVariableOp!assignvariableop_57_conv2d_8_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58Ј
AssignVariableOp_58AssignVariableOp/assignvariableop_58_batch_normalization_8_gammaIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59ґ
AssignVariableOp_59AssignVariableOp.assignvariableop_59_batch_normalization_8_betaIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60љ
AssignVariableOp_60AssignVariableOp5assignvariableop_60_batch_normalization_8_moving_meanIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Ѕ
AssignVariableOp_61AssignVariableOp9assignvariableop_61_batch_normalization_8_moving_varianceIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Њ
AssignVariableOp_62AssignVariableOp6assignvariableop_62_batch_normalization_11_moving_meanIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63¬
AssignVariableOp_63AssignVariableOp:assignvariableop_63_batch_normalization_11_moving_varianceIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64°
AssignVariableOp_64AssignVariableOpassignvariableop_64_totalIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65°
AssignVariableOp_65AssignVariableOpassignvariableop_65_countIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66£
AssignVariableOp_66AssignVariableOpassignvariableop_66_total_1Identity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67£
AssignVariableOp_67AssignVariableOpassignvariableop_67_count_1Identity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68≥
AssignVariableOp_68AssignVariableOp+assignvariableop_68_adam_conv2d_11_kernel_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69±
AssignVariableOp_69AssignVariableOp)assignvariableop_69_adam_conv2d_11_bias_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70њ
AssignVariableOp_70AssignVariableOp7assignvariableop_70_adam_batch_normalization_11_gamma_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Њ
AssignVariableOp_71AssignVariableOp6assignvariableop_71_adam_batch_normalization_11_beta_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72≥
AssignVariableOp_72AssignVariableOp+assignvariableop_72_adam_conv2d_11_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73±
AssignVariableOp_73AssignVariableOp)assignvariableop_73_adam_conv2d_11_bias_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74њ
AssignVariableOp_74AssignVariableOp7assignvariableop_74_adam_batch_normalization_11_gamma_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75Њ
AssignVariableOp_75AssignVariableOp6assignvariableop_75_adam_batch_normalization_11_beta_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_759
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpж
Identity_76Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_76ў
Identity_77IdentityIdentity_76:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_77"#
identity_77Identity_77:output:0*«
_input_shapesµ
≤: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
√
ц
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34131929

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34126918

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34131480

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
с

c
G__inference_reshape_1_layer_call_and_return_conditional_losses_34127726

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
strided_slice/stack_2в
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
€€€€€€€€€2
Reshape/shape/1Ж
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34125291

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34131418

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34127365

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
≠G
С
E__inference_model_1_layer_call_and_return_conditional_losses_34128593

inputs
input_conv_34128458
input_conv_34128460
input_conv_34128462
input_conv_34128464
input_conv_34128466
input_conv_34128468
input_conv_34128470
input_conv_34128472
input_conv_34128474
input_conv_34128476
input_conv_34128478
input_conv_34128480
res_block_0_34128483
res_block_0_34128485
res_block_0_34128487
res_block_0_34128489
res_block_0_34128491
res_block_0_34128493
res_block_0_34128495
res_block_0_34128497
res_block_0_34128499
res_block_0_34128501
res_block_0_34128503
res_block_0_34128505
res_block_0_34128507
res_block_0_34128509
res_block_0_34128511
res_block_0_34128513
res_block_0_34128515
res_block_0_34128517
res_block_1_34128522
res_block_1_34128524
res_block_1_34128526
res_block_1_34128528
res_block_1_34128530
res_block_1_34128532
res_block_1_34128534
res_block_1_34128536
res_block_1_34128538
res_block_1_34128540
res_block_1_34128542
res_block_1_34128544
res_block_1_34128546
res_block_1_34128548
res_block_1_34128550
res_block_1_34128552
res_block_1_34128554
res_block_1_34128556
feature_34128561
feature_34128563
feature_34128565
feature_34128567
feature_34128569
feature_34128571
output_conv_34128577
output_conv_34128579
output_conv_34128581
output_conv_34128583
output_conv_34128585
output_conv_34128587
identityИҐfeature/StatefulPartitionedCallҐ"input_conv/StatefulPartitionedCallҐ#output_conv/StatefulPartitionedCallҐ#res_block_0/StatefulPartitionedCallҐ#res_block_1/StatefulPartitionedCallЧ
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_34128458input_conv_34128460input_conv_34128462input_conv_34128464input_conv_34128466input_conv_34128468input_conv_34128470input_conv_34128472input_conv_34128474input_conv_34128476input_conv_34128478input_conv_34128480*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_341254122$
"input_conv/StatefulPartitionedCallџ
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_34128483res_block_0_34128485res_block_0_34128487res_block_0_34128489res_block_0_34128491res_block_0_34128493res_block_0_34128495res_block_0_34128497res_block_0_34128499res_block_0_34128501res_block_0_34128503res_block_0_34128505res_block_0_34128507res_block_0_34128509res_block_0_34128511res_block_0_34128513res_block_0_34128515res_block_0_34128517*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_341262232%
#res_block_0/StatefulPartitionedCall÷
tf.__operators__.add/AddV2AddV2+input_conv/StatefulPartitionedCall:output:0,res_block_0/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
tf.__operators__.add/AddV2х
relu_0/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_relu_0_layer_call_and_return_conditional_losses_341280392
relu_0/PartitionedCallѕ
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_34128522res_block_1_34128524res_block_1_34128526res_block_1_34128528res_block_1_34128530res_block_1_34128532res_block_1_34128534res_block_1_34128536res_block_1_34128538res_block_1_34128540res_block_1_34128542res_block_1_34128544res_block_1_34128546res_block_1_34128548res_block_1_34128550res_block_1_34128552res_block_1_34128554res_block_1_34128556*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_341270722%
#res_block_1/StatefulPartitionedCallќ
tf.__operators__.add_1/AddV2AddV2relu_0/PartitionedCall:output:0,res_block_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
tf.__operators__.add_1/AddV2ч
relu_1/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_relu_1_layer_call_and_return_conditional_losses_341281722
relu_1/PartitionedCallЛ
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_34128561feature_34128563feature_34128565feature_34128567feature_34128569feature_34128571*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_341274562!
feature/StatefulPartitionedCallІ
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/permе
 tf.compat.v1.transpose/transpose	Transpose(feature/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:€€€€€€€€€R2"
 tf.compat.v1.transpose/transposeУ
feature_linear/PartitionedCallPartitionedCall$tf.compat.v1.transpose/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_feature_linear_layer_call_and_return_conditional_losses_341282332 
feature_linear/PartitionedCall≠
#output_conv/StatefulPartitionedCallStatefulPartitionedCall'feature_linear/PartitionedCall:output:0output_conv_34128577output_conv_34128579output_conv_34128581output_conv_34128583output_conv_34128585output_conv_34128587*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_output_conv_layer_call_and_return_conditional_losses_341277762%
#output_conv/StatefulPartitionedCallР
output_linear/PartitionedCallPartitionedCall,output_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_output_linear_layer_call_and_return_conditional_losses_341282922
output_linear/PartitionedCallш
softmax/PartitionedCallPartitionedCall&output_linear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_341283052
softmax/PartitionedCall≠
IdentityIdentity softmax/PartitionedCall:output:0 ^feature/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall$^res_block_0/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*°
_input_shapesП
М:€€€€€€€€€ь::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
feature/StatefulPartitionedCallfeature/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#output_conv/StatefulPartitionedCall#output_conv/StatefulPartitionedCall2J
#res_block_0/StatefulPartitionedCall#res_block_0/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ь
 
_user_specified_nameinputs
Ў
f
J__inference_activation_6_layer_call_and_return_conditional_losses_34127406

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
Ж
А
+__inference_conv2d_4_layer_call_fn_34131257

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_341260362
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ХЦ
БA
#__inference__wrapped_model_34124916	
input<
8model_1_input_conv_conv2d_conv2d_readvariableop_resource=
9model_1_input_conv_conv2d_biasadd_readvariableop_resourceB
>model_1_input_conv_batch_normalization_readvariableop_resourceD
@model_1_input_conv_batch_normalization_readvariableop_1_resourceS
Omodel_1_input_conv_batch_normalization_fusedbatchnormv3_readvariableop_resourceU
Qmodel_1_input_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource>
:model_1_input_conv_conv2d_1_conv2d_readvariableop_resource?
;model_1_input_conv_conv2d_1_biasadd_readvariableop_resourceD
@model_1_input_conv_batch_normalization_1_readvariableop_resourceF
Bmodel_1_input_conv_batch_normalization_1_readvariableop_1_resourceU
Qmodel_1_input_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceW
Smodel_1_input_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource?
;model_1_res_block_0_conv2d_2_conv2d_readvariableop_resource@
<model_1_res_block_0_conv2d_2_biasadd_readvariableop_resourceE
Amodel_1_res_block_0_batch_normalization_2_readvariableop_resourceG
Cmodel_1_res_block_0_batch_normalization_2_readvariableop_1_resourceV
Rmodel_1_res_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceX
Tmodel_1_res_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource?
;model_1_res_block_0_conv2d_3_conv2d_readvariableop_resource@
<model_1_res_block_0_conv2d_3_biasadd_readvariableop_resourceE
Amodel_1_res_block_0_batch_normalization_3_readvariableop_resourceG
Cmodel_1_res_block_0_batch_normalization_3_readvariableop_1_resourceV
Rmodel_1_res_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceX
Tmodel_1_res_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource?
;model_1_res_block_0_conv2d_4_conv2d_readvariableop_resource@
<model_1_res_block_0_conv2d_4_biasadd_readvariableop_resourceE
Amodel_1_res_block_0_batch_normalization_4_readvariableop_resourceG
Cmodel_1_res_block_0_batch_normalization_4_readvariableop_1_resourceV
Rmodel_1_res_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceX
Tmodel_1_res_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource?
;model_1_res_block_1_conv2d_5_conv2d_readvariableop_resource@
<model_1_res_block_1_conv2d_5_biasadd_readvariableop_resourceE
Amodel_1_res_block_1_batch_normalization_5_readvariableop_resourceG
Cmodel_1_res_block_1_batch_normalization_5_readvariableop_1_resourceV
Rmodel_1_res_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceX
Tmodel_1_res_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource?
;model_1_res_block_1_conv2d_6_conv2d_readvariableop_resource@
<model_1_res_block_1_conv2d_6_biasadd_readvariableop_resourceE
Amodel_1_res_block_1_batch_normalization_6_readvariableop_resourceG
Cmodel_1_res_block_1_batch_normalization_6_readvariableop_1_resourceV
Rmodel_1_res_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceX
Tmodel_1_res_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource?
;model_1_res_block_1_conv2d_7_conv2d_readvariableop_resource@
<model_1_res_block_1_conv2d_7_biasadd_readvariableop_resourceE
Amodel_1_res_block_1_batch_normalization_7_readvariableop_resourceG
Cmodel_1_res_block_1_batch_normalization_7_readvariableop_1_resourceV
Rmodel_1_res_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceX
Tmodel_1_res_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource;
7model_1_feature_conv2d_8_conv2d_readvariableop_resource<
8model_1_feature_conv2d_8_biasadd_readvariableop_resourceA
=model_1_feature_batch_normalization_8_readvariableop_resourceC
?model_1_feature_batch_normalization_8_readvariableop_1_resourceR
Nmodel_1_feature_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceT
Pmodel_1_feature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource@
<model_1_output_conv_conv2d_11_conv2d_readvariableop_resourceA
=model_1_output_conv_conv2d_11_biasadd_readvariableop_resourceF
Bmodel_1_output_conv_batch_normalization_11_readvariableop_resourceH
Dmodel_1_output_conv_batch_normalization_11_readvariableop_1_resourceW
Smodel_1_output_conv_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceY
Umodel_1_output_conv_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource
identityИҐEmodel_1/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpҐGmodel_1/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ґ4model_1/feature/batch_normalization_8/ReadVariableOpҐ6model_1/feature/batch_normalization_8/ReadVariableOp_1Ґ/model_1/feature/conv2d_8/BiasAdd/ReadVariableOpҐ.model_1/feature/conv2d_8/Conv2D/ReadVariableOpҐFmodel_1/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpҐHmodel_1/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ5model_1/input_conv/batch_normalization/ReadVariableOpҐ7model_1/input_conv/batch_normalization/ReadVariableOp_1ҐHmodel_1/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐJmodel_1/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ7model_1/input_conv/batch_normalization_1/ReadVariableOpҐ9model_1/input_conv/batch_normalization_1/ReadVariableOp_1Ґ0model_1/input_conv/conv2d/BiasAdd/ReadVariableOpҐ/model_1/input_conv/conv2d/Conv2D/ReadVariableOpҐ2model_1/input_conv/conv2d_1/BiasAdd/ReadVariableOpҐ1model_1/input_conv/conv2d_1/Conv2D/ReadVariableOpҐJmodel_1/output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpҐLmodel_1/output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Ґ9model_1/output_conv/batch_normalization_11/ReadVariableOpҐ;model_1/output_conv/batch_normalization_11/ReadVariableOp_1Ґ4model_1/output_conv/conv2d_11/BiasAdd/ReadVariableOpҐ3model_1/output_conv/conv2d_11/Conv2D/ReadVariableOpҐImodel_1/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐKmodel_1/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ8model_1/res_block_0/batch_normalization_2/ReadVariableOpҐ:model_1/res_block_0/batch_normalization_2/ReadVariableOp_1ҐImodel_1/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐKmodel_1/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ8model_1/res_block_0/batch_normalization_3/ReadVariableOpҐ:model_1/res_block_0/batch_normalization_3/ReadVariableOp_1ҐImodel_1/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpҐKmodel_1/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ґ8model_1/res_block_0/batch_normalization_4/ReadVariableOpҐ:model_1/res_block_0/batch_normalization_4/ReadVariableOp_1Ґ3model_1/res_block_0/conv2d_2/BiasAdd/ReadVariableOpҐ2model_1/res_block_0/conv2d_2/Conv2D/ReadVariableOpҐ3model_1/res_block_0/conv2d_3/BiasAdd/ReadVariableOpҐ2model_1/res_block_0/conv2d_3/Conv2D/ReadVariableOpҐ3model_1/res_block_0/conv2d_4/BiasAdd/ReadVariableOpҐ2model_1/res_block_0/conv2d_4/Conv2D/ReadVariableOpҐImodel_1/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpҐKmodel_1/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ґ8model_1/res_block_1/batch_normalization_5/ReadVariableOpҐ:model_1/res_block_1/batch_normalization_5/ReadVariableOp_1ҐImodel_1/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpҐKmodel_1/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ґ8model_1/res_block_1/batch_normalization_6/ReadVariableOpҐ:model_1/res_block_1/batch_normalization_6/ReadVariableOp_1ҐImodel_1/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpҐKmodel_1/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ґ8model_1/res_block_1/batch_normalization_7/ReadVariableOpҐ:model_1/res_block_1/batch_normalization_7/ReadVariableOp_1Ґ3model_1/res_block_1/conv2d_5/BiasAdd/ReadVariableOpҐ2model_1/res_block_1/conv2d_5/Conv2D/ReadVariableOpҐ3model_1/res_block_1/conv2d_6/BiasAdd/ReadVariableOpҐ2model_1/res_block_1/conv2d_6/Conv2D/ReadVariableOpҐ3model_1/res_block_1/conv2d_7/BiasAdd/ReadVariableOpҐ2model_1/res_block_1/conv2d_7/Conv2D/ReadVariableOpд
/model_1/input_conv/conv2d/Conv2D/ReadVariableOpReadVariableOp8model_1_input_conv_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype021
/model_1/input_conv/conv2d/Conv2D/ReadVariableOpу
 model_1/input_conv/conv2d/Conv2DConv2Dinput7model_1/input_conv/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€®А*
paddingVALID*
strides
2"
 model_1/input_conv/conv2d/Conv2Dџ
0model_1/input_conv/conv2d/BiasAdd/ReadVariableOpReadVariableOp9model_1_input_conv_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0model_1/input_conv/conv2d/BiasAdd/ReadVariableOpт
!model_1/input_conv/conv2d/BiasAddBiasAdd)model_1/input_conv/conv2d/Conv2D:output:08model_1/input_conv/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€®А2#
!model_1/input_conv/conv2d/BiasAddк
5model_1/input_conv/batch_normalization/ReadVariableOpReadVariableOp>model_1_input_conv_batch_normalization_readvariableop_resource*
_output_shapes	
:А*
dtype027
5model_1/input_conv/batch_normalization/ReadVariableOpр
7model_1/input_conv/batch_normalization/ReadVariableOp_1ReadVariableOp@model_1_input_conv_batch_normalization_readvariableop_1_resource*
_output_shapes	
:А*
dtype029
7model_1/input_conv/batch_normalization/ReadVariableOp_1Э
Fmodel_1/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpOmodel_1_input_conv_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02H
Fmodel_1/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp£
Hmodel_1/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQmodel_1_input_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02J
Hmodel_1/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ё
7model_1/input_conv/batch_normalization/FusedBatchNormV3FusedBatchNormV3*model_1/input_conv/conv2d/BiasAdd:output:0=model_1/input_conv/batch_normalization/ReadVariableOp:value:0?model_1/input_conv/batch_normalization/ReadVariableOp_1:value:0Nmodel_1/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Pmodel_1/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:€€€€€€€€€®А:А:А:А:А:*
epsilon%oГ:*
is_training( 29
7model_1/input_conv/batch_normalization/FusedBatchNormV3…
"model_1/input_conv/activation/ReluRelu;model_1/input_conv/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:€€€€€€€€€®А2$
"model_1/input_conv/activation/Reluк
1model_1/input_conv/conv2d_1/Conv2D/ReadVariableOpReadVariableOp:model_1_input_conv_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype023
1model_1/input_conv/conv2d_1/Conv2D/ReadVariableOpҐ
"model_1/input_conv/conv2d_1/Conv2DConv2D0model_1/input_conv/activation/Relu:activations:09model_1/input_conv/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingVALID*
strides
2$
"model_1/input_conv/conv2d_1/Conv2Dа
2model_1/input_conv/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp;model_1_input_conv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype024
2model_1/input_conv/conv2d_1/BiasAdd/ReadVariableOpш
#model_1/input_conv/conv2d_1/BiasAddBiasAdd+model_1/input_conv/conv2d_1/Conv2D:output:0:model_1/input_conv/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2%
#model_1/input_conv/conv2d_1/BiasAddп
7model_1/input_conv/batch_normalization_1/ReadVariableOpReadVariableOp@model_1_input_conv_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype029
7model_1/input_conv/batch_normalization_1/ReadVariableOpх
9model_1/input_conv/batch_normalization_1/ReadVariableOp_1ReadVariableOpBmodel_1_input_conv_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9model_1/input_conv/batch_normalization_1/ReadVariableOp_1Ґ
Hmodel_1/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpQmodel_1_input_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02J
Hmodel_1/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp®
Jmodel_1/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpSmodel_1_input_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02L
Jmodel_1/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ж
9model_1/input_conv/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3,model_1/input_conv/conv2d_1/BiasAdd:output:0?model_1/input_conv/batch_normalization_1/ReadVariableOp:value:0Amodel_1/input_conv/batch_normalization_1/ReadVariableOp_1:value:0Pmodel_1/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Rmodel_1/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2;
9model_1/input_conv/batch_normalization_1/FusedBatchNormV3Ќ
$model_1/input_conv/activation_1/ReluRelu=model_1/input_conv/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2&
$model_1/input_conv/activation_1/Reluм
2model_1/res_block_0/conv2d_2/Conv2D/ReadVariableOpReadVariableOp;model_1_res_block_0_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2model_1/res_block_0/conv2d_2/Conv2D/ReadVariableOp¶
#model_1/res_block_0/conv2d_2/Conv2DConv2D2model_1/input_conv/activation_1/Relu:activations:0:model_1/res_block_0/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2%
#model_1/res_block_0/conv2d_2/Conv2Dг
3model_1/res_block_0/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp<model_1_res_block_0_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3model_1/res_block_0/conv2d_2/BiasAdd/ReadVariableOpь
$model_1/res_block_0/conv2d_2/BiasAddBiasAdd,model_1/res_block_0/conv2d_2/Conv2D:output:0;model_1/res_block_0/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2&
$model_1/res_block_0/conv2d_2/BiasAddт
8model_1/res_block_0/batch_normalization_2/ReadVariableOpReadVariableOpAmodel_1_res_block_0_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/res_block_0/batch_normalization_2/ReadVariableOpш
:model_1/res_block_0/batch_normalization_2/ReadVariableOp_1ReadVariableOpCmodel_1_res_block_0_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:model_1/res_block_0/batch_normalization_2/ReadVariableOp_1•
Imodel_1/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpRmodel_1_res_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02K
Imodel_1/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpЂ
Kmodel_1/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTmodel_1_res_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02M
Kmodel_1/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1н
:model_1/res_block_0/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3-model_1/res_block_0/conv2d_2/BiasAdd:output:0@model_1/res_block_0/batch_normalization_2/ReadVariableOp:value:0Bmodel_1/res_block_0/batch_normalization_2/ReadVariableOp_1:value:0Qmodel_1/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Smodel_1/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2<
:model_1/res_block_0/batch_normalization_2/FusedBatchNormV3–
%model_1/res_block_0/activation_2/ReluRelu>model_1/res_block_0/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2'
%model_1/res_block_0/activation_2/Reluм
2model_1/res_block_0/conv2d_3/Conv2D/ReadVariableOpReadVariableOp;model_1_res_block_0_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2model_1/res_block_0/conv2d_3/Conv2D/ReadVariableOpІ
#model_1/res_block_0/conv2d_3/Conv2DConv2D3model_1/res_block_0/activation_2/Relu:activations:0:model_1/res_block_0/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2%
#model_1/res_block_0/conv2d_3/Conv2Dг
3model_1/res_block_0/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp<model_1_res_block_0_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3model_1/res_block_0/conv2d_3/BiasAdd/ReadVariableOpь
$model_1/res_block_0/conv2d_3/BiasAddBiasAdd,model_1/res_block_0/conv2d_3/Conv2D:output:0;model_1/res_block_0/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2&
$model_1/res_block_0/conv2d_3/BiasAddт
8model_1/res_block_0/batch_normalization_3/ReadVariableOpReadVariableOpAmodel_1_res_block_0_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/res_block_0/batch_normalization_3/ReadVariableOpш
:model_1/res_block_0/batch_normalization_3/ReadVariableOp_1ReadVariableOpCmodel_1_res_block_0_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:model_1/res_block_0/batch_normalization_3/ReadVariableOp_1•
Imodel_1/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpRmodel_1_res_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02K
Imodel_1/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpЂ
Kmodel_1/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTmodel_1_res_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02M
Kmodel_1/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1н
:model_1/res_block_0/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3-model_1/res_block_0/conv2d_3/BiasAdd:output:0@model_1/res_block_0/batch_normalization_3/ReadVariableOp:value:0Bmodel_1/res_block_0/batch_normalization_3/ReadVariableOp_1:value:0Qmodel_1/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Smodel_1/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2<
:model_1/res_block_0/batch_normalization_3/FusedBatchNormV3–
%model_1/res_block_0/activation_3/ReluRelu>model_1/res_block_0/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2'
%model_1/res_block_0/activation_3/Reluм
2model_1/res_block_0/conv2d_4/Conv2D/ReadVariableOpReadVariableOp;model_1_res_block_0_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2model_1/res_block_0/conv2d_4/Conv2D/ReadVariableOpІ
#model_1/res_block_0/conv2d_4/Conv2DConv2D3model_1/res_block_0/activation_3/Relu:activations:0:model_1/res_block_0/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2%
#model_1/res_block_0/conv2d_4/Conv2Dг
3model_1/res_block_0/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp<model_1_res_block_0_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3model_1/res_block_0/conv2d_4/BiasAdd/ReadVariableOpь
$model_1/res_block_0/conv2d_4/BiasAddBiasAdd,model_1/res_block_0/conv2d_4/Conv2D:output:0;model_1/res_block_0/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2&
$model_1/res_block_0/conv2d_4/BiasAddт
8model_1/res_block_0/batch_normalization_4/ReadVariableOpReadVariableOpAmodel_1_res_block_0_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/res_block_0/batch_normalization_4/ReadVariableOpш
:model_1/res_block_0/batch_normalization_4/ReadVariableOp_1ReadVariableOpCmodel_1_res_block_0_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:model_1/res_block_0/batch_normalization_4/ReadVariableOp_1•
Imodel_1/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpRmodel_1_res_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02K
Imodel_1/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpЂ
Kmodel_1/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTmodel_1_res_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02M
Kmodel_1/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1н
:model_1/res_block_0/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3-model_1/res_block_0/conv2d_4/BiasAdd:output:0@model_1/res_block_0/batch_normalization_4/ReadVariableOp:value:0Bmodel_1/res_block_0/batch_normalization_4/ReadVariableOp_1:value:0Qmodel_1/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Smodel_1/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2<
:model_1/res_block_0/batch_normalization_4/FusedBatchNormV3€
"model_1/tf.__operators__.add/AddV2AddV22model_1/input_conv/activation_1/Relu:activations:0>model_1/res_block_0/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2$
"model_1/tf.__operators__.add/AddV2Ф
model_1/relu_0/ReluRelu&model_1/tf.__operators__.add/AddV2:z:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
model_1/relu_0/Reluм
2model_1/res_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp;model_1_res_block_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2model_1/res_block_1/conv2d_5/Conv2D/ReadVariableOpХ
#model_1/res_block_1/conv2d_5/Conv2DConv2D!model_1/relu_0/Relu:activations:0:model_1/res_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2%
#model_1/res_block_1/conv2d_5/Conv2Dг
3model_1/res_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp<model_1_res_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3model_1/res_block_1/conv2d_5/BiasAdd/ReadVariableOpь
$model_1/res_block_1/conv2d_5/BiasAddBiasAdd,model_1/res_block_1/conv2d_5/Conv2D:output:0;model_1/res_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2&
$model_1/res_block_1/conv2d_5/BiasAddт
8model_1/res_block_1/batch_normalization_5/ReadVariableOpReadVariableOpAmodel_1_res_block_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/res_block_1/batch_normalization_5/ReadVariableOpш
:model_1/res_block_1/batch_normalization_5/ReadVariableOp_1ReadVariableOpCmodel_1_res_block_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:model_1/res_block_1/batch_normalization_5/ReadVariableOp_1•
Imodel_1/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpRmodel_1_res_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02K
Imodel_1/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpЂ
Kmodel_1/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTmodel_1_res_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02M
Kmodel_1/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1н
:model_1/res_block_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3-model_1/res_block_1/conv2d_5/BiasAdd:output:0@model_1/res_block_1/batch_normalization_5/ReadVariableOp:value:0Bmodel_1/res_block_1/batch_normalization_5/ReadVariableOp_1:value:0Qmodel_1/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Smodel_1/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2<
:model_1/res_block_1/batch_normalization_5/FusedBatchNormV3–
%model_1/res_block_1/activation_4/ReluRelu>model_1/res_block_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2'
%model_1/res_block_1/activation_4/Reluм
2model_1/res_block_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp;model_1_res_block_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2model_1/res_block_1/conv2d_6/Conv2D/ReadVariableOpІ
#model_1/res_block_1/conv2d_6/Conv2DConv2D3model_1/res_block_1/activation_4/Relu:activations:0:model_1/res_block_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2%
#model_1/res_block_1/conv2d_6/Conv2Dг
3model_1/res_block_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp<model_1_res_block_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3model_1/res_block_1/conv2d_6/BiasAdd/ReadVariableOpь
$model_1/res_block_1/conv2d_6/BiasAddBiasAdd,model_1/res_block_1/conv2d_6/Conv2D:output:0;model_1/res_block_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2&
$model_1/res_block_1/conv2d_6/BiasAddт
8model_1/res_block_1/batch_normalization_6/ReadVariableOpReadVariableOpAmodel_1_res_block_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/res_block_1/batch_normalization_6/ReadVariableOpш
:model_1/res_block_1/batch_normalization_6/ReadVariableOp_1ReadVariableOpCmodel_1_res_block_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:model_1/res_block_1/batch_normalization_6/ReadVariableOp_1•
Imodel_1/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpRmodel_1_res_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02K
Imodel_1/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpЂ
Kmodel_1/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTmodel_1_res_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02M
Kmodel_1/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1н
:model_1/res_block_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3-model_1/res_block_1/conv2d_6/BiasAdd:output:0@model_1/res_block_1/batch_normalization_6/ReadVariableOp:value:0Bmodel_1/res_block_1/batch_normalization_6/ReadVariableOp_1:value:0Qmodel_1/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Smodel_1/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2<
:model_1/res_block_1/batch_normalization_6/FusedBatchNormV3–
%model_1/res_block_1/activation_5/ReluRelu>model_1/res_block_1/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2'
%model_1/res_block_1/activation_5/Reluм
2model_1/res_block_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp;model_1_res_block_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2model_1/res_block_1/conv2d_7/Conv2D/ReadVariableOpІ
#model_1/res_block_1/conv2d_7/Conv2DConv2D3model_1/res_block_1/activation_5/Relu:activations:0:model_1/res_block_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2%
#model_1/res_block_1/conv2d_7/Conv2Dг
3model_1/res_block_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp<model_1_res_block_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3model_1/res_block_1/conv2d_7/BiasAdd/ReadVariableOpь
$model_1/res_block_1/conv2d_7/BiasAddBiasAdd,model_1/res_block_1/conv2d_7/Conv2D:output:0;model_1/res_block_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2&
$model_1/res_block_1/conv2d_7/BiasAddт
8model_1/res_block_1/batch_normalization_7/ReadVariableOpReadVariableOpAmodel_1_res_block_1_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02:
8model_1/res_block_1/batch_normalization_7/ReadVariableOpш
:model_1/res_block_1/batch_normalization_7/ReadVariableOp_1ReadVariableOpCmodel_1_res_block_1_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:model_1/res_block_1/batch_normalization_7/ReadVariableOp_1•
Imodel_1/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpRmodel_1_res_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02K
Imodel_1/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЂ
Kmodel_1/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTmodel_1_res_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02M
Kmodel_1/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1н
:model_1/res_block_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3-model_1/res_block_1/conv2d_7/BiasAdd:output:0@model_1/res_block_1/batch_normalization_7/ReadVariableOp:value:0Bmodel_1/res_block_1/batch_normalization_7/ReadVariableOp_1:value:0Qmodel_1/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Smodel_1/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2<
:model_1/res_block_1/batch_normalization_7/FusedBatchNormV3т
$model_1/tf.__operators__.add_1/AddV2AddV2!model_1/relu_0/Relu:activations:0>model_1/res_block_1/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2&
$model_1/tf.__operators__.add_1/AddV2Ц
model_1/relu_1/ReluRelu(model_1/tf.__operators__.add_1/AddV2:z:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
model_1/relu_1/Reluа
.model_1/feature/conv2d_8/Conv2D/ReadVariableOpReadVariableOp7model_1_feature_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype020
.model_1/feature/conv2d_8/Conv2D/ReadVariableOpК
model_1/feature/conv2d_8/Conv2DConv2D!model_1/relu_1/Relu:activations:06model_1/feature/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R*
paddingVALID*
strides
2!
model_1/feature/conv2d_8/Conv2D„
/model_1/feature/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp8model_1_feature_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/model_1/feature/conv2d_8/BiasAdd/ReadVariableOpм
 model_1/feature/conv2d_8/BiasAddBiasAdd(model_1/feature/conv2d_8/Conv2D:output:07model_1/feature/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R2"
 model_1/feature/conv2d_8/BiasAddж
4model_1/feature/batch_normalization_8/ReadVariableOpReadVariableOp=model_1_feature_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype026
4model_1/feature/batch_normalization_8/ReadVariableOpм
6model_1/feature/batch_normalization_8/ReadVariableOp_1ReadVariableOp?model_1_feature_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype028
6model_1/feature/batch_normalization_8/ReadVariableOp_1Щ
Emodel_1/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpNmodel_1_feature_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02G
Emodel_1/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpЯ
Gmodel_1/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPmodel_1_feature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02I
Gmodel_1/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1—
6model_1/feature/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3)model_1/feature/conv2d_8/BiasAdd:output:0<model_1/feature/batch_normalization_8/ReadVariableOp:value:0>model_1/feature/batch_normalization_8/ReadVariableOp_1:value:0Mmodel_1/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Omodel_1/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R:::::*
epsilon%oГ:*
is_training( 28
6model_1/feature/batch_normalization_8/FusedBatchNormV3ƒ
!model_1/feature/activation_6/ReluRelu:model_1/feature/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R2#
!model_1/feature/activation_6/ReluЈ
-model_1/tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-model_1/tf.compat.v1.transpose/transpose/permД
(model_1/tf.compat.v1.transpose/transpose	Transpose/model_1/feature/activation_6/Relu:activations:06model_1/tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:€€€€€€€€€R2*
(model_1/tf.compat.v1.transpose/transposeп
3model_1/output_conv/conv2d_11/Conv2D/ReadVariableOpReadVariableOp<model_1_output_conv_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype025
3model_1/output_conv/conv2d_11/Conv2D/ReadVariableOp§
$model_1/output_conv/conv2d_11/Conv2DConv2D,model_1/tf.compat.v1.transpose/transpose:y:0;model_1/output_conv/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2&
$model_1/output_conv/conv2d_11/Conv2Dж
4model_1/output_conv/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp=model_1_output_conv_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4model_1/output_conv/conv2d_11/BiasAdd/ReadVariableOpА
%model_1/output_conv/conv2d_11/BiasAddBiasAdd-model_1/output_conv/conv2d_11/Conv2D:output:0<model_1/output_conv/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2'
%model_1/output_conv/conv2d_11/BiasAddх
9model_1/output_conv/batch_normalization_11/ReadVariableOpReadVariableOpBmodel_1_output_conv_batch_normalization_11_readvariableop_resource*
_output_shapes
:*
dtype02;
9model_1/output_conv/batch_normalization_11/ReadVariableOpы
;model_1/output_conv/batch_normalization_11/ReadVariableOp_1ReadVariableOpDmodel_1_output_conv_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;model_1/output_conv/batch_normalization_11/ReadVariableOp_1®
Jmodel_1/output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpSmodel_1_output_conv_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jmodel_1/output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpЃ
Lmodel_1/output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpUmodel_1_output_conv_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02N
Lmodel_1/output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ф
;model_1/output_conv/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3.model_1/output_conv/conv2d_11/BiasAdd:output:0Amodel_1/output_conv/batch_normalization_11/ReadVariableOp:value:0Cmodel_1/output_conv/batch_normalization_11/ReadVariableOp_1:value:0Rmodel_1/output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Tmodel_1/output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2=
;model_1/output_conv/batch_normalization_11/FusedBatchNormV3є
#model_1/output_conv/reshape_1/ShapeShape?model_1/output_conv/batch_normalization_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2%
#model_1/output_conv/reshape_1/Shape∞
1model_1/output_conv/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_1/output_conv/reshape_1/strided_slice/stackі
3model_1/output_conv/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model_1/output_conv/reshape_1/strided_slice/stack_1і
3model_1/output_conv/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_1/output_conv/reshape_1/strided_slice/stack_2Ц
+model_1/output_conv/reshape_1/strided_sliceStridedSlice,model_1/output_conv/reshape_1/Shape:output:0:model_1/output_conv/reshape_1/strided_slice/stack:output:0<model_1/output_conv/reshape_1/strided_slice/stack_1:output:0<model_1/output_conv/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model_1/output_conv/reshape_1/strided_slice©
-model_1/output_conv/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-model_1/output_conv/reshape_1/Reshape/shape/1ю
+model_1/output_conv/reshape_1/Reshape/shapePack4model_1/output_conv/reshape_1/strided_slice:output:06model_1/output_conv/reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+model_1/output_conv/reshape_1/Reshape/shapeВ
%model_1/output_conv/reshape_1/ReshapeReshape?model_1/output_conv/batch_normalization_11/FusedBatchNormV3:y:04model_1/output_conv/reshape_1/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2'
%model_1/output_conv/reshape_1/ReshapeЯ
model_1/softmax/SoftmaxSoftmax.model_1/output_conv/reshape_1/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_1/softmax/Softmax≠
IdentityIdentity!model_1/softmax/Softmax:softmax:0F^model_1/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpH^model_1/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_15^model_1/feature/batch_normalization_8/ReadVariableOp7^model_1/feature/batch_normalization_8/ReadVariableOp_10^model_1/feature/conv2d_8/BiasAdd/ReadVariableOp/^model_1/feature/conv2d_8/Conv2D/ReadVariableOpG^model_1/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpI^model_1/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_16^model_1/input_conv/batch_normalization/ReadVariableOp8^model_1/input_conv/batch_normalization/ReadVariableOp_1I^model_1/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpK^model_1/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_18^model_1/input_conv/batch_normalization_1/ReadVariableOp:^model_1/input_conv/batch_normalization_1/ReadVariableOp_11^model_1/input_conv/conv2d/BiasAdd/ReadVariableOp0^model_1/input_conv/conv2d/Conv2D/ReadVariableOp3^model_1/input_conv/conv2d_1/BiasAdd/ReadVariableOp2^model_1/input_conv/conv2d_1/Conv2D/ReadVariableOpK^model_1/output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpM^model_1/output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:^model_1/output_conv/batch_normalization_11/ReadVariableOp<^model_1/output_conv/batch_normalization_11/ReadVariableOp_15^model_1/output_conv/conv2d_11/BiasAdd/ReadVariableOp4^model_1/output_conv/conv2d_11/Conv2D/ReadVariableOpJ^model_1/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpL^model_1/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_19^model_1/res_block_0/batch_normalization_2/ReadVariableOp;^model_1/res_block_0/batch_normalization_2/ReadVariableOp_1J^model_1/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpL^model_1/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_19^model_1/res_block_0/batch_normalization_3/ReadVariableOp;^model_1/res_block_0/batch_normalization_3/ReadVariableOp_1J^model_1/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpL^model_1/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_19^model_1/res_block_0/batch_normalization_4/ReadVariableOp;^model_1/res_block_0/batch_normalization_4/ReadVariableOp_14^model_1/res_block_0/conv2d_2/BiasAdd/ReadVariableOp3^model_1/res_block_0/conv2d_2/Conv2D/ReadVariableOp4^model_1/res_block_0/conv2d_3/BiasAdd/ReadVariableOp3^model_1/res_block_0/conv2d_3/Conv2D/ReadVariableOp4^model_1/res_block_0/conv2d_4/BiasAdd/ReadVariableOp3^model_1/res_block_0/conv2d_4/Conv2D/ReadVariableOpJ^model_1/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpL^model_1/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_19^model_1/res_block_1/batch_normalization_5/ReadVariableOp;^model_1/res_block_1/batch_normalization_5/ReadVariableOp_1J^model_1/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpL^model_1/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_19^model_1/res_block_1/batch_normalization_6/ReadVariableOp;^model_1/res_block_1/batch_normalization_6/ReadVariableOp_1J^model_1/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpL^model_1/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_19^model_1/res_block_1/batch_normalization_7/ReadVariableOp;^model_1/res_block_1/batch_normalization_7/ReadVariableOp_14^model_1/res_block_1/conv2d_5/BiasAdd/ReadVariableOp3^model_1/res_block_1/conv2d_5/Conv2D/ReadVariableOp4^model_1/res_block_1/conv2d_6/BiasAdd/ReadVariableOp3^model_1/res_block_1/conv2d_6/Conv2D/ReadVariableOp4^model_1/res_block_1/conv2d_7/BiasAdd/ReadVariableOp3^model_1/res_block_1/conv2d_7/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*°
_input_shapesП
М:€€€€€€€€€ь::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2О
Emodel_1/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpEmodel_1/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2Т
Gmodel_1/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Gmodel_1/feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12l
4model_1/feature/batch_normalization_8/ReadVariableOp4model_1/feature/batch_normalization_8/ReadVariableOp2p
6model_1/feature/batch_normalization_8/ReadVariableOp_16model_1/feature/batch_normalization_8/ReadVariableOp_12b
/model_1/feature/conv2d_8/BiasAdd/ReadVariableOp/model_1/feature/conv2d_8/BiasAdd/ReadVariableOp2`
.model_1/feature/conv2d_8/Conv2D/ReadVariableOp.model_1/feature/conv2d_8/Conv2D/ReadVariableOp2Р
Fmodel_1/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpFmodel_1/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp2Ф
Hmodel_1/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Hmodel_1/input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_12n
5model_1/input_conv/batch_normalization/ReadVariableOp5model_1/input_conv/batch_normalization/ReadVariableOp2r
7model_1/input_conv/batch_normalization/ReadVariableOp_17model_1/input_conv/batch_normalization/ReadVariableOp_12Ф
Hmodel_1/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpHmodel_1/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2Ш
Jmodel_1/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Jmodel_1/input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12r
7model_1/input_conv/batch_normalization_1/ReadVariableOp7model_1/input_conv/batch_normalization_1/ReadVariableOp2v
9model_1/input_conv/batch_normalization_1/ReadVariableOp_19model_1/input_conv/batch_normalization_1/ReadVariableOp_12d
0model_1/input_conv/conv2d/BiasAdd/ReadVariableOp0model_1/input_conv/conv2d/BiasAdd/ReadVariableOp2b
/model_1/input_conv/conv2d/Conv2D/ReadVariableOp/model_1/input_conv/conv2d/Conv2D/ReadVariableOp2h
2model_1/input_conv/conv2d_1/BiasAdd/ReadVariableOp2model_1/input_conv/conv2d_1/BiasAdd/ReadVariableOp2f
1model_1/input_conv/conv2d_1/Conv2D/ReadVariableOp1model_1/input_conv/conv2d_1/Conv2D/ReadVariableOp2Ш
Jmodel_1/output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpJmodel_1/output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2Ь
Lmodel_1/output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Lmodel_1/output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12v
9model_1/output_conv/batch_normalization_11/ReadVariableOp9model_1/output_conv/batch_normalization_11/ReadVariableOp2z
;model_1/output_conv/batch_normalization_11/ReadVariableOp_1;model_1/output_conv/batch_normalization_11/ReadVariableOp_12l
4model_1/output_conv/conv2d_11/BiasAdd/ReadVariableOp4model_1/output_conv/conv2d_11/BiasAdd/ReadVariableOp2j
3model_1/output_conv/conv2d_11/Conv2D/ReadVariableOp3model_1/output_conv/conv2d_11/Conv2D/ReadVariableOp2Ц
Imodel_1/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpImodel_1/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2Ъ
Kmodel_1/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Kmodel_1/res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12t
8model_1/res_block_0/batch_normalization_2/ReadVariableOp8model_1/res_block_0/batch_normalization_2/ReadVariableOp2x
:model_1/res_block_0/batch_normalization_2/ReadVariableOp_1:model_1/res_block_0/batch_normalization_2/ReadVariableOp_12Ц
Imodel_1/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpImodel_1/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2Ъ
Kmodel_1/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Kmodel_1/res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12t
8model_1/res_block_0/batch_normalization_3/ReadVariableOp8model_1/res_block_0/batch_normalization_3/ReadVariableOp2x
:model_1/res_block_0/batch_normalization_3/ReadVariableOp_1:model_1/res_block_0/batch_normalization_3/ReadVariableOp_12Ц
Imodel_1/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpImodel_1/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2Ъ
Kmodel_1/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Kmodel_1/res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12t
8model_1/res_block_0/batch_normalization_4/ReadVariableOp8model_1/res_block_0/batch_normalization_4/ReadVariableOp2x
:model_1/res_block_0/batch_normalization_4/ReadVariableOp_1:model_1/res_block_0/batch_normalization_4/ReadVariableOp_12j
3model_1/res_block_0/conv2d_2/BiasAdd/ReadVariableOp3model_1/res_block_0/conv2d_2/BiasAdd/ReadVariableOp2h
2model_1/res_block_0/conv2d_2/Conv2D/ReadVariableOp2model_1/res_block_0/conv2d_2/Conv2D/ReadVariableOp2j
3model_1/res_block_0/conv2d_3/BiasAdd/ReadVariableOp3model_1/res_block_0/conv2d_3/BiasAdd/ReadVariableOp2h
2model_1/res_block_0/conv2d_3/Conv2D/ReadVariableOp2model_1/res_block_0/conv2d_3/Conv2D/ReadVariableOp2j
3model_1/res_block_0/conv2d_4/BiasAdd/ReadVariableOp3model_1/res_block_0/conv2d_4/BiasAdd/ReadVariableOp2h
2model_1/res_block_0/conv2d_4/Conv2D/ReadVariableOp2model_1/res_block_0/conv2d_4/Conv2D/ReadVariableOp2Ц
Imodel_1/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpImodel_1/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2Ъ
Kmodel_1/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Kmodel_1/res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12t
8model_1/res_block_1/batch_normalization_5/ReadVariableOp8model_1/res_block_1/batch_normalization_5/ReadVariableOp2x
:model_1/res_block_1/batch_normalization_5/ReadVariableOp_1:model_1/res_block_1/batch_normalization_5/ReadVariableOp_12Ц
Imodel_1/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpImodel_1/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2Ъ
Kmodel_1/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Kmodel_1/res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12t
8model_1/res_block_1/batch_normalization_6/ReadVariableOp8model_1/res_block_1/batch_normalization_6/ReadVariableOp2x
:model_1/res_block_1/batch_normalization_6/ReadVariableOp_1:model_1/res_block_1/batch_normalization_6/ReadVariableOp_12Ц
Imodel_1/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpImodel_1/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2Ъ
Kmodel_1/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Kmodel_1/res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12t
8model_1/res_block_1/batch_normalization_7/ReadVariableOp8model_1/res_block_1/batch_normalization_7/ReadVariableOp2x
:model_1/res_block_1/batch_normalization_7/ReadVariableOp_1:model_1/res_block_1/batch_normalization_7/ReadVariableOp_12j
3model_1/res_block_1/conv2d_5/BiasAdd/ReadVariableOp3model_1/res_block_1/conv2d_5/BiasAdd/ReadVariableOp2h
2model_1/res_block_1/conv2d_5/Conv2D/ReadVariableOp2model_1/res_block_1/conv2d_5/Conv2D/ReadVariableOp2j
3model_1/res_block_1/conv2d_6/BiasAdd/ReadVariableOp3model_1/res_block_1/conv2d_6/BiasAdd/ReadVariableOp2h
2model_1/res_block_1/conv2d_6/Conv2D/ReadVariableOp2model_1/res_block_1/conv2d_6/Conv2D/ReadVariableOp2j
3model_1/res_block_1/conv2d_7/BiasAdd/ReadVariableOp3model_1/res_block_1/conv2d_7/BiasAdd/ReadVariableOp2h
2model_1/res_block_1/conv2d_7/Conv2D/ReadVariableOp2model_1/res_block_1/conv2d_7/Conv2D/ReadVariableOp:W S
0
_output_shapes
:€€€€€€€€€ь

_user_specified_nameinput
ї
a
E__inference_softmax_layer_call_and_return_conditional_losses_34130621

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:€€€€€€€€€2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
а
Ђ
8__inference_batch_normalization_6_layer_call_fn_34131677

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_341268262
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
®
Ђ
8__inference_batch_normalization_5_layer_call_fn_34131462

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_341264402
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
И
Ы
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34132022

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ў
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3≠
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ю
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Й
ф
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34125163

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1–
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:€€€€€€€€€®А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3№
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:€€€€€€€€€®А2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:€€€€€€€€€®А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:€€€€€€€€€®А
 
_user_specified_nameinputs
л
э
E__inference_feature_layer_call_and_return_conditional_losses_34127434
conv2d_8_input
conv2d_8_34127418
conv2d_8_34127420"
batch_normalization_8_34127423"
batch_normalization_8_34127425"
batch_normalization_8_34127427"
batch_normalization_8_34127429
identityИҐ-batch_normalization_8/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallѓ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_34127418conv2d_8_34127420*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_8_layer_call_and_return_conditional_losses_341273142"
 conv2d_8/StatefulPartitionedCallѕ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_34127423batch_normalization_8_34127425batch_normalization_8_34127427batch_normalization_8_34127429*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_341273652/
-batch_normalization_8/StatefulPartitionedCallЯ
activation_6/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_6_layer_call_and_return_conditional_losses_341274062
activation_6/PartitionedCall‘
IdentityIdentity%activation_6/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R@::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€R@
(
_user_specified_nameconv2d_8_input
а
Ђ
8__inference_batch_normalization_4_layer_call_fn_34131381

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_341260872
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
д
©
6__inference_batch_normalization_layer_call_fn_34130756

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_341251632
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€®А2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:€€€€€€€€€®А::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€®А
 
_user_specified_nameinputs
„
Б
I__inference_output_conv_layer_call_and_return_conditional_losses_34127812

inputs
conv2d_11_34127796
conv2d_11_34127798#
batch_normalization_11_34127801#
batch_normalization_11_34127803#
batch_normalization_11_34127805#
batch_normalization_11_34127807
identityИҐ.batch_normalization_11/StatefulPartitionedCallҐ!conv2d_11/StatefulPartitionedCallђ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_11_34127796conv2d_11_34127798*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_341276252#
!conv2d_11/StatefulPartitionedCall„
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_11_34127801batch_normalization_11_34127803batch_normalization_11_34127805batch_normalization_11_34127807*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *]
fXRV
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3412767820
.batch_normalization_11/StatefulPartitionedCallП
reshape_1/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_341277262
reshape_1/PartitionedCallЋ
IdentityIdentity"reshape_1/PartitionedCall:output:0/^batch_normalization_11/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R::::::2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
“
`
D__inference_relu_0_layer_call_and_return_conditional_losses_34128039

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ё
g
K__inference_output_linear_layer_call_and_return_conditional_losses_34128292

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34130878

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
э!
щ
H__inference_input_conv_layer_call_and_return_conditional_losses_34125412

inputs
conv2d_34125381
conv2d_34125383 
batch_normalization_34125386 
batch_normalization_34125388 
batch_normalization_34125390 
batch_normalization_34125392
conv2d_1_34125396
conv2d_1_34125398"
batch_normalization_1_34125401"
batch_normalization_1_34125403"
batch_normalization_1_34125405"
batch_normalization_1_34125407
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallЯ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_34125381conv2d_34125383*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_341251302 
conv2d/StatefulPartitionedCallЅ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_34125386batch_normalization_34125388batch_normalization_34125390batch_normalization_34125392*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_341251632-
+batch_normalization/StatefulPartitionedCallЩ
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_341252222
activation/PartitionedCallƒ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_34125396conv2d_1_34125398*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_341252402"
 conv2d_1/StatefulPartitionedCallѕ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_34125401batch_normalization_1_34125403batch_normalization_1_34125405batch_normalization_1_34125407*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_341252732/
-batch_normalization_1/StatefulPartitionedCallЯ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_341253322
activation_1/PartitionedCall£
IdentityIdentity%activation_1/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€ь::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ь
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34131786

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
®
Ђ
8__inference_batch_normalization_2_layer_call_fn_34131075

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_341255912
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34126069

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ч
»
.__inference_output_conv_layer_call_fn_34127827
conv2d_11_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallconv2d_11_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_output_conv_layer_call_and_return_conditional_losses_341278122
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:€€€€€€€€€R
)
_user_specified_nameconv2d_11_input
®
Ђ
8__inference_batch_normalization_7_layer_call_fn_34131817

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_341266092
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34125959

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
х	
Ё
D__inference_conv2d_layer_call_and_return_conditional_losses_34130636

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€®А*
paddingVALID*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€®А2	
BiasAddЯ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€®А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€ь::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€ь
 
_user_specified_nameinputs
≤
В
.__inference_res_block_0_layer_call_fn_34126351
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
identityИҐStatefulPartitionedCallе
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
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_341263122
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€R@
(
_user_specified_nameconv2d_2_input
Ў
f
J__inference_activation_2_layer_call_and_return_conditional_losses_34131080

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34127347

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
™
ђ
9__inference_batch_normalization_11_layer_call_fn_34132130

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *]
fXRV
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_341276002
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34125105

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
И
Ы
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34127660

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ў
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3≠
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1ю
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34126936

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ѓG
С
E__inference_model_1_layer_call_and_return_conditional_losses_34128856

inputs
input_conv_34128721
input_conv_34128723
input_conv_34128725
input_conv_34128727
input_conv_34128729
input_conv_34128731
input_conv_34128733
input_conv_34128735
input_conv_34128737
input_conv_34128739
input_conv_34128741
input_conv_34128743
res_block_0_34128746
res_block_0_34128748
res_block_0_34128750
res_block_0_34128752
res_block_0_34128754
res_block_0_34128756
res_block_0_34128758
res_block_0_34128760
res_block_0_34128762
res_block_0_34128764
res_block_0_34128766
res_block_0_34128768
res_block_0_34128770
res_block_0_34128772
res_block_0_34128774
res_block_0_34128776
res_block_0_34128778
res_block_0_34128780
res_block_1_34128785
res_block_1_34128787
res_block_1_34128789
res_block_1_34128791
res_block_1_34128793
res_block_1_34128795
res_block_1_34128797
res_block_1_34128799
res_block_1_34128801
res_block_1_34128803
res_block_1_34128805
res_block_1_34128807
res_block_1_34128809
res_block_1_34128811
res_block_1_34128813
res_block_1_34128815
res_block_1_34128817
res_block_1_34128819
feature_34128824
feature_34128826
feature_34128828
feature_34128830
feature_34128832
feature_34128834
output_conv_34128840
output_conv_34128842
output_conv_34128844
output_conv_34128846
output_conv_34128848
output_conv_34128850
identityИҐfeature/StatefulPartitionedCallҐ"input_conv/StatefulPartitionedCallҐ#output_conv/StatefulPartitionedCallҐ#res_block_0/StatefulPartitionedCallҐ#res_block_1/StatefulPartitionedCallЧ
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_34128721input_conv_34128723input_conv_34128725input_conv_34128727input_conv_34128729input_conv_34128731input_conv_34128733input_conv_34128735input_conv_34128737input_conv_34128739input_conv_34128741input_conv_34128743*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_341254752$
"input_conv/StatefulPartitionedCallџ
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_34128746res_block_0_34128748res_block_0_34128750res_block_0_34128752res_block_0_34128754res_block_0_34128756res_block_0_34128758res_block_0_34128760res_block_0_34128762res_block_0_34128764res_block_0_34128766res_block_0_34128768res_block_0_34128770res_block_0_34128772res_block_0_34128774res_block_0_34128776res_block_0_34128778res_block_0_34128780*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_341263122%
#res_block_0/StatefulPartitionedCall÷
tf.__operators__.add/AddV2AddV2+input_conv/StatefulPartitionedCall:output:0,res_block_0/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
tf.__operators__.add/AddV2х
relu_0/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_relu_0_layer_call_and_return_conditional_losses_341280392
relu_0/PartitionedCallѕ
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_34128785res_block_1_34128787res_block_1_34128789res_block_1_34128791res_block_1_34128793res_block_1_34128795res_block_1_34128797res_block_1_34128799res_block_1_34128801res_block_1_34128803res_block_1_34128805res_block_1_34128807res_block_1_34128809res_block_1_34128811res_block_1_34128813res_block_1_34128815res_block_1_34128817res_block_1_34128819*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_341271612%
#res_block_1/StatefulPartitionedCallќ
tf.__operators__.add_1/AddV2AddV2relu_0/PartitionedCall:output:0,res_block_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
tf.__operators__.add_1/AddV2ч
relu_1/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_relu_1_layer_call_and_return_conditional_losses_341281722
relu_1/PartitionedCallЛ
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_34128824feature_34128826feature_34128828feature_34128830feature_34128832feature_34128834*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_341274922!
feature/StatefulPartitionedCallІ
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/permе
 tf.compat.v1.transpose/transpose	Transpose(feature/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:€€€€€€€€€R2"
 tf.compat.v1.transpose/transposeУ
feature_linear/PartitionedCallPartitionedCall$tf.compat.v1.transpose/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_feature_linear_layer_call_and_return_conditional_losses_341282332 
feature_linear/PartitionedCallѓ
#output_conv/StatefulPartitionedCallStatefulPartitionedCall'feature_linear/PartitionedCall:output:0output_conv_34128840output_conv_34128842output_conv_34128844output_conv_34128846output_conv_34128848output_conv_34128850*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_output_conv_layer_call_and_return_conditional_losses_341278122%
#output_conv/StatefulPartitionedCallР
output_linear/PartitionedCallPartitionedCall,output_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_output_linear_layer_call_and_return_conditional_losses_341282922
output_linear/PartitionedCallш
softmax/PartitionedCallPartitionedCall&output_linear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_341283052
softmax/PartitionedCall≠
IdentityIdentity softmax/PartitionedCall:output:0 ^feature/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall$^res_block_0/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*°
_input_shapesП
М:€€€€€€€€€ь::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
feature/StatefulPartitionedCallfeature/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#output_conv/StatefulPartitionedCall#output_conv/StatefulPartitionedCall2J
#res_block_0/StatefulPartitionedCall#res_block_0/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ь
 
_user_specified_nameinputs
х	
Ё
D__inference_conv2d_layer_call_and_return_conditional_losses_34125130

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€®А*
paddingVALID*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€®А2	
BiasAddЯ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€®А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€ь::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€ь
 
_user_specified_nameinputs
т
С
&__inference_signature_wrapper_34129112	
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

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58
identityИҐStatefulPartitionedCallъ
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
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*^
_read_only_resource_inputs@
><	
 !"#$%&'()*+,-./0123456789:;<*2
config_proto" 

CPU

GPU2 *0J 8В *,
f'R%
#__inference__wrapped_model_341249162
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*°
_input_shapesП
М:€€€€€€€€€ь::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:€€€€€€€€€ь

_user_specified_nameinput
Ж
А
+__inference_conv2d_7_layer_call_fn_34131706

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_7_layer_call_and_return_conditional_losses_341268852
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
а
Ђ
8__inference_batch_normalization_3_layer_call_fn_34131228

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_341259772
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ъ
њ
.__inference_output_conv_layer_call_fn_34130590

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCall∞
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_output_conv_layer_call_and_return_conditional_losses_341277762
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
а	
Я
-__inference_input_conv_layer_call_fn_34125439
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
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_341254122
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€ь::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:€€€€€€€€€ь
&
_user_specified_nameconv2d_input
®
Ђ
8__inference_batch_normalization_2_layer_call_fn_34131062

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_341255602
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34131337

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ќ	
Щ
-__inference_input_conv_layer_call_fn_34129937

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
identityИҐStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_341254122
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€ь::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ь
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34130987

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
®
ђ
9__inference_batch_normalization_11_layer_call_fn_34132117

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *]
fXRV
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_341275692
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34131571

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Д
ї
*__inference_feature_layer_call_fn_34130498

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCallґ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_341274922
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R@::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ьV
ё
I__inference_res_block_1_layer_call_and_return_conditional_losses_34130322

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
identityИҐ5batch_normalization_5/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_5/ReadVariableOpҐ&batch_normalization_5/ReadVariableOp_1Ґ5batch_normalization_6/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_6/ReadVariableOpҐ&batch_normalization_6/ReadVariableOp_1Ґ5batch_normalization_7/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_7/ReadVariableOpҐ&batch_normalization_7/ReadVariableOp_1Ґconv2d_5/BiasAdd/ReadVariableOpҐconv2d_5/Conv2D/ReadVariableOpҐconv2d_6/BiasAdd/ReadVariableOpҐconv2d_6/Conv2D/ReadVariableOpҐconv2d_7/BiasAdd/ReadVariableOpҐconv2d_7/Conv2D/ReadVariableOp∞
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOpЊ
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
conv2d_5/Conv2DІ
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOpђ
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
conv2d_5/BiasAddґ
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOpЉ
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1й
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3Ф
activation_4/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
activation_4/Relu∞
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_6/Conv2D/ReadVariableOp„
conv2d_6/Conv2DConv2Dactivation_4/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
conv2d_6/Conv2DІ
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOpђ
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
conv2d_6/BiasAddґ
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOpЉ
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1й
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3Ф
activation_5/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
activation_5/Relu∞
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp„
conv2d_7/Conv2DConv2Dactivation_5/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
conv2d_7/Conv2DІ
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOpђ
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
conv2d_7/BiasAddґ
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_7/ReadVariableOpЉ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_7/ReadVariableOp_1й
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3Х
IdentityIdentity*batch_normalization_7/FusedBatchNormV3:y:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::2n
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
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
K
/__inference_activation_2_layer_call_fn_34131085

inputs
identity’
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_341259082
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34126409

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ц(
∆
I__inference_output_conv_layer_call_and_return_conditional_losses_34130573

inputs,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource
identityИҐ6batch_normalization_11/FusedBatchNormV3/ReadVariableOpҐ8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Ґ%batch_normalization_11/ReadVariableOpҐ'batch_normalization_11/ReadVariableOp_1Ґ conv2d_11/BiasAdd/ReadVariableOpҐconv2d_11/Conv2D/ReadVariableOp≥
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02!
conv2d_11/Conv2D/ReadVariableOp¬
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv2d_11/Conv2D™
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp∞
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_11/BiasAddє
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_11/ReadVariableOpњ
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_11/ReadVariableOp_1м
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpт
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1и
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2)
'batch_normalization_11/FusedBatchNormV3}
reshape_1/ShapeShape+batch_normalization_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
reshape_1/ShapeИ
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stackМ
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1М
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2Ю
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_sliceБ
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
reshape_1/Reshape/shape/1Ѓ
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape≤
reshape_1/ReshapeReshape+batch_normalization_11/FusedBatchNormV3:y:0 reshape_1/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
reshape_1/Reshapeщ
IdentityIdentityreshape_1/Reshape:output:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R::::::2p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
Щ
Ц
*__inference_model_1_layer_call_fn_34129691

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

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58
identityИҐStatefulPartitionedCallЫ	
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
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_341285932
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*°
_input_shapesП
М:€€€€€€€€€ь::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ь
 
_user_specified_nameinputs
а	
Я
-__inference_input_conv_layer_call_fn_34125502
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
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_341254752
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€ь::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:€€€€€€€€€ь
&
_user_specified_nameconv2d_input
ы
ц
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34126826

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
®
Ђ
8__inference_batch_normalization_1_layer_call_fn_34130847

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_341250742
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ь
ч
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34127678

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ќ
ф
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34130681

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3н
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34131049

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34127258

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
н	
я
F__inference_conv2d_8_layer_call_and_return_conditional_losses_34131840

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34126716

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
®
Ђ
8__inference_batch_normalization_4_layer_call_fn_34131306

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_341257602
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
а
Ђ
8__inference_batch_normalization_1_layer_call_fn_34130922

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_341252912
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ў
f
J__inference_activation_2_layer_call_and_return_conditional_losses_34125908

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
“
`
D__inference_relu_1_layer_call_and_return_conditional_losses_34128172

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ц
Х
*__inference_model_1_layer_call_fn_34128716	
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

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58
identityИҐStatefulPartitionedCallЪ	
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
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*\
_read_only_resource_inputs>
<:	
 !"#$%&'()*+,-./0123456789:*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_341285932
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*°
_input_shapesП
М:€€€€€€€€€ь::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:€€€€€€€€€ь

_user_specified_nameinput
√
ц
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34131293

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
л
э
E__inference_feature_layer_call_and_return_conditional_losses_34127415
conv2d_8_input
conv2d_8_34127325
conv2d_8_34127327"
batch_normalization_8_34127392"
batch_normalization_8_34127394"
batch_normalization_8_34127396"
batch_normalization_8_34127398
identityИҐ-batch_normalization_8/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallѓ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_34127325conv2d_8_34127327*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_8_layer_call_and_return_conditional_losses_341273142"
 conv2d_8/StatefulPartitionedCallѕ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_34127392batch_normalization_8_34127394batch_normalization_8_34127396batch_normalization_8_34127398*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_341273472/
-batch_normalization_8/StatefulPartitionedCallЯ
activation_6/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_6_layer_call_and_return_conditional_losses_341274062
activation_6/PartitionedCall‘
IdentityIdentity%activation_6/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R@::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€R@
(
_user_specified_nameconv2d_8_input
К
А
+__inference_conv2d_1_layer_call_fn_34130798

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_341252402
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€®А::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€®А
 
_user_specified_nameinputs
√
K
/__inference_activation_4_layer_call_fn_34131534

inputs
identity’
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_4_layer_call_and_return_conditional_losses_341267572
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Іц
ќ:
E__inference_model_1_layer_call_and_return_conditional_losses_34129340

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
Hfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource8
4output_conv_conv2d_11_conv2d_readvariableop_resource9
5output_conv_conv2d_11_biasadd_readvariableop_resource>
:output_conv_batch_normalization_11_readvariableop_resource@
<output_conv_batch_normalization_11_readvariableop_1_resourceO
Koutput_conv_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceQ
Moutput_conv_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource
identityИҐ=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpҐ?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ґ,feature/batch_normalization_8/ReadVariableOpҐ.feature/batch_normalization_8/ReadVariableOp_1Ґ'feature/conv2d_8/BiasAdd/ReadVariableOpҐ&feature/conv2d_8/Conv2D/ReadVariableOpҐ>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpҐ@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ-input_conv/batch_normalization/ReadVariableOpҐ/input_conv/batch_normalization/ReadVariableOp_1Ґ@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐBinput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ/input_conv/batch_normalization_1/ReadVariableOpҐ1input_conv/batch_normalization_1/ReadVariableOp_1Ґ(input_conv/conv2d/BiasAdd/ReadVariableOpҐ'input_conv/conv2d/Conv2D/ReadVariableOpҐ*input_conv/conv2d_1/BiasAdd/ReadVariableOpҐ)input_conv/conv2d_1/Conv2D/ReadVariableOpҐ1output_conv/batch_normalization_11/AssignNewValueҐ3output_conv/batch_normalization_11/AssignNewValue_1ҐBoutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpҐDoutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Ґ1output_conv/batch_normalization_11/ReadVariableOpҐ3output_conv/batch_normalization_11/ReadVariableOp_1Ґ,output_conv/conv2d_11/BiasAdd/ReadVariableOpҐ+output_conv/conv2d_11/Conv2D/ReadVariableOpҐAres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐCres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ0res_block_0/batch_normalization_2/ReadVariableOpҐ2res_block_0/batch_normalization_2/ReadVariableOp_1ҐAres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐCres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ0res_block_0/batch_normalization_3/ReadVariableOpҐ2res_block_0/batch_normalization_3/ReadVariableOp_1ҐAres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpҐCres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ґ0res_block_0/batch_normalization_4/ReadVariableOpҐ2res_block_0/batch_normalization_4/ReadVariableOp_1Ґ+res_block_0/conv2d_2/BiasAdd/ReadVariableOpҐ*res_block_0/conv2d_2/Conv2D/ReadVariableOpҐ+res_block_0/conv2d_3/BiasAdd/ReadVariableOpҐ*res_block_0/conv2d_3/Conv2D/ReadVariableOpҐ+res_block_0/conv2d_4/BiasAdd/ReadVariableOpҐ*res_block_0/conv2d_4/Conv2D/ReadVariableOpҐAres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpҐCres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ґ0res_block_1/batch_normalization_5/ReadVariableOpҐ2res_block_1/batch_normalization_5/ReadVariableOp_1ҐAres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpҐCres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ґ0res_block_1/batch_normalization_6/ReadVariableOpҐ2res_block_1/batch_normalization_6/ReadVariableOp_1ҐAres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpҐCres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ґ0res_block_1/batch_normalization_7/ReadVariableOpҐ2res_block_1/batch_normalization_7/ReadVariableOp_1Ґ+res_block_1/conv2d_5/BiasAdd/ReadVariableOpҐ*res_block_1/conv2d_5/Conv2D/ReadVariableOpҐ+res_block_1/conv2d_6/BiasAdd/ReadVariableOpҐ*res_block_1/conv2d_6/Conv2D/ReadVariableOpҐ+res_block_1/conv2d_7/BiasAdd/ReadVariableOpҐ*res_block_1/conv2d_7/Conv2D/ReadVariableOpћ
'input_conv/conv2d/Conv2D/ReadVariableOpReadVariableOp0input_conv_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02)
'input_conv/conv2d/Conv2D/ReadVariableOp№
input_conv/conv2d/Conv2DConv2Dinputs/input_conv/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€®А*
paddingVALID*
strides
2
input_conv/conv2d/Conv2D√
(input_conv/conv2d/BiasAdd/ReadVariableOpReadVariableOp1input_conv_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(input_conv/conv2d/BiasAdd/ReadVariableOp“
input_conv/conv2d/BiasAddBiasAdd!input_conv/conv2d/Conv2D:output:00input_conv/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€®А2
input_conv/conv2d/BiasAdd“
-input_conv/batch_normalization/ReadVariableOpReadVariableOp6input_conv_batch_normalization_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-input_conv/batch_normalization/ReadVariableOpЎ
/input_conv/batch_normalization/ReadVariableOp_1ReadVariableOp8input_conv_batch_normalization_readvariableop_1_resource*
_output_shapes	
:А*
dtype021
/input_conv/batch_normalization/ReadVariableOp_1Е
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02@
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpЛ
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02B
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¶
/input_conv/batch_normalization/FusedBatchNormV3FusedBatchNormV3"input_conv/conv2d/BiasAdd:output:05input_conv/batch_normalization/ReadVariableOp:value:07input_conv/batch_normalization/ReadVariableOp_1:value:0Finput_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hinput_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:€€€€€€€€€®А:А:А:А:А:*
epsilon%oГ:*
is_training( 21
/input_conv/batch_normalization/FusedBatchNormV3±
input_conv/activation/ReluRelu3input_conv/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:€€€€€€€€€®А2
input_conv/activation/Relu“
)input_conv/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2input_conv_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype02+
)input_conv/conv2d_1/Conv2D/ReadVariableOpВ
input_conv/conv2d_1/Conv2DConv2D(input_conv/activation/Relu:activations:01input_conv/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingVALID*
strides
2
input_conv/conv2d_1/Conv2D»
*input_conv/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3input_conv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*input_conv/conv2d_1/BiasAdd/ReadVariableOpЎ
input_conv/conv2d_1/BiasAddBiasAdd#input_conv/conv2d_1/Conv2D:output:02input_conv/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
input_conv/conv2d_1/BiasAdd„
/input_conv/batch_normalization_1/ReadVariableOpReadVariableOp8input_conv_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype021
/input_conv/batch_normalization_1/ReadVariableOpЁ
1input_conv/batch_normalization_1/ReadVariableOp_1ReadVariableOp:input_conv_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1input_conv/batch_normalization_1/ReadVariableOp_1К
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02B
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpР
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ѓ
1input_conv/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$input_conv/conv2d_1/BiasAdd:output:07input_conv/batch_normalization_1/ReadVariableOp:value:09input_conv/batch_normalization_1/ReadVariableOp_1:value:0Hinput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jinput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 23
1input_conv/batch_normalization_1/FusedBatchNormV3µ
input_conv/activation_1/ReluRelu5input_conv/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
input_conv/activation_1/Relu‘
*res_block_0/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_2/Conv2D/ReadVariableOpЖ
res_block_0/conv2d_2/Conv2DConv2D*input_conv/activation_1/Relu:activations:02res_block_0/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
res_block_0/conv2d_2/Conv2DЋ
+res_block_0/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_2/BiasAdd/ReadVariableOp№
res_block_0/conv2d_2/BiasAddBiasAdd$res_block_0/conv2d_2/Conv2D:output:03res_block_0/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_0/conv2d_2/BiasAddЏ
0res_block_0/batch_normalization_2/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_2/ReadVariableOpа
2res_block_0/batch_normalization_2/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_2/ReadVariableOp_1Н
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpУ
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1µ
2res_block_0/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_2/BiasAdd:output:08res_block_0/batch_normalization_2/ReadVariableOp:value:0:res_block_0/batch_normalization_2/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 24
2res_block_0/batch_normalization_2/FusedBatchNormV3Є
res_block_0/activation_2/ReluRelu6res_block_0/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_0/activation_2/Relu‘
*res_block_0/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_3/Conv2D/ReadVariableOpЗ
res_block_0/conv2d_3/Conv2DConv2D+res_block_0/activation_2/Relu:activations:02res_block_0/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
res_block_0/conv2d_3/Conv2DЋ
+res_block_0/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_3/BiasAdd/ReadVariableOp№
res_block_0/conv2d_3/BiasAddBiasAdd$res_block_0/conv2d_3/Conv2D:output:03res_block_0/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_0/conv2d_3/BiasAddЏ
0res_block_0/batch_normalization_3/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_3/ReadVariableOpа
2res_block_0/batch_normalization_3/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_3/ReadVariableOp_1Н
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpУ
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1µ
2res_block_0/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_3/BiasAdd:output:08res_block_0/batch_normalization_3/ReadVariableOp:value:0:res_block_0/batch_normalization_3/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 24
2res_block_0/batch_normalization_3/FusedBatchNormV3Є
res_block_0/activation_3/ReluRelu6res_block_0/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_0/activation_3/Relu‘
*res_block_0/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_4/Conv2D/ReadVariableOpЗ
res_block_0/conv2d_4/Conv2DConv2D+res_block_0/activation_3/Relu:activations:02res_block_0/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
res_block_0/conv2d_4/Conv2DЋ
+res_block_0/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_4/BiasAdd/ReadVariableOp№
res_block_0/conv2d_4/BiasAddBiasAdd$res_block_0/conv2d_4/Conv2D:output:03res_block_0/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_0/conv2d_4/BiasAddЏ
0res_block_0/batch_normalization_4/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_4/ReadVariableOpа
2res_block_0/batch_normalization_4/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_4/ReadVariableOp_1Н
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpУ
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1µ
2res_block_0/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_4/BiasAdd:output:08res_block_0/batch_normalization_4/ReadVariableOp:value:0:res_block_0/batch_normalization_4/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 24
2res_block_0/batch_normalization_4/FusedBatchNormV3я
tf.__operators__.add/AddV2AddV2*input_conv/activation_1/Relu:activations:06res_block_0/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
tf.__operators__.add/AddV2|
relu_0/ReluRelutf.__operators__.add/AddV2:z:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
relu_0/Relu‘
*res_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_5/Conv2D/ReadVariableOpх
res_block_1/conv2d_5/Conv2DConv2Drelu_0/Relu:activations:02res_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
res_block_1/conv2d_5/Conv2DЋ
+res_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_5/BiasAdd/ReadVariableOp№
res_block_1/conv2d_5/BiasAddBiasAdd$res_block_1/conv2d_5/Conv2D:output:03res_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_1/conv2d_5/BiasAddЏ
0res_block_1/batch_normalization_5/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_5/ReadVariableOpа
2res_block_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_5/ReadVariableOp_1Н
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpУ
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1µ
2res_block_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_5/BiasAdd:output:08res_block_1/batch_normalization_5/ReadVariableOp:value:0:res_block_1/batch_normalization_5/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 24
2res_block_1/batch_normalization_5/FusedBatchNormV3Є
res_block_1/activation_4/ReluRelu6res_block_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_1/activation_4/Relu‘
*res_block_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_6/Conv2D/ReadVariableOpЗ
res_block_1/conv2d_6/Conv2DConv2D+res_block_1/activation_4/Relu:activations:02res_block_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
res_block_1/conv2d_6/Conv2DЋ
+res_block_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_6/BiasAdd/ReadVariableOp№
res_block_1/conv2d_6/BiasAddBiasAdd$res_block_1/conv2d_6/Conv2D:output:03res_block_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_1/conv2d_6/BiasAddЏ
0res_block_1/batch_normalization_6/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_6/ReadVariableOpа
2res_block_1/batch_normalization_6/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_6/ReadVariableOp_1Н
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpУ
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1µ
2res_block_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_6/BiasAdd:output:08res_block_1/batch_normalization_6/ReadVariableOp:value:0:res_block_1/batch_normalization_6/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 24
2res_block_1/batch_normalization_6/FusedBatchNormV3Є
res_block_1/activation_5/ReluRelu6res_block_1/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_1/activation_5/Relu‘
*res_block_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_7/Conv2D/ReadVariableOpЗ
res_block_1/conv2d_7/Conv2DConv2D+res_block_1/activation_5/Relu:activations:02res_block_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
res_block_1/conv2d_7/Conv2DЋ
+res_block_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_7/BiasAdd/ReadVariableOp№
res_block_1/conv2d_7/BiasAddBiasAdd$res_block_1/conv2d_7/Conv2D:output:03res_block_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_1/conv2d_7/BiasAddЏ
0res_block_1/batch_normalization_7/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_7/ReadVariableOpа
2res_block_1/batch_normalization_7/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_7/ReadVariableOp_1Н
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpУ
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1µ
2res_block_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_7/BiasAdd:output:08res_block_1/batch_normalization_7/ReadVariableOp:value:0:res_block_1/batch_normalization_7/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 24
2res_block_1/batch_normalization_7/FusedBatchNormV3“
tf.__operators__.add_1/AddV2AddV2relu_0/Relu:activations:06res_block_1/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
tf.__operators__.add_1/AddV2~
relu_1/ReluRelu tf.__operators__.add_1/AddV2:z:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
relu_1/Relu»
&feature/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/feature_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&feature/conv2d_8/Conv2D/ReadVariableOpк
feature/conv2d_8/Conv2DConv2Drelu_1/Relu:activations:0.feature/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R*
paddingVALID*
strides
2
feature/conv2d_8/Conv2Dњ
'feature/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0feature_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'feature/conv2d_8/BiasAdd/ReadVariableOpћ
feature/conv2d_8/BiasAddBiasAdd feature/conv2d_8/Conv2D:output:0/feature/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R2
feature/conv2d_8/BiasAddќ
,feature/batch_normalization_8/ReadVariableOpReadVariableOp5feature_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02.
,feature/batch_normalization_8/ReadVariableOp‘
.feature/batch_normalization_8/ReadVariableOp_1ReadVariableOp7feature_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype020
.feature/batch_normalization_8/ReadVariableOp_1Б
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpЗ
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Щ
.feature/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3!feature/conv2d_8/BiasAdd:output:04feature/batch_normalization_8/ReadVariableOp:value:06feature/batch_normalization_8/ReadVariableOp_1:value:0Efeature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gfeature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R:::::*
epsilon%oГ:*
is_training( 20
.feature/batch_normalization_8/FusedBatchNormV3ђ
feature/activation_6/ReluRelu2feature/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R2
feature/activation_6/ReluІ
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/permд
 tf.compat.v1.transpose/transpose	Transpose'feature/activation_6/Relu:activations:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:€€€€€€€€€R2"
 tf.compat.v1.transpose/transpose„
+output_conv/conv2d_11/Conv2D/ReadVariableOpReadVariableOp4output_conv_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02-
+output_conv/conv2d_11/Conv2D/ReadVariableOpД
output_conv/conv2d_11/Conv2DConv2D$tf.compat.v1.transpose/transpose:y:03output_conv/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
output_conv/conv2d_11/Conv2Dќ
,output_conv/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp5output_conv_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,output_conv/conv2d_11/BiasAdd/ReadVariableOpа
output_conv/conv2d_11/BiasAddBiasAdd%output_conv/conv2d_11/Conv2D:output:04output_conv/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
output_conv/conv2d_11/BiasAddЁ
1output_conv/batch_normalization_11/ReadVariableOpReadVariableOp:output_conv_batch_normalization_11_readvariableop_resource*
_output_shapes
:*
dtype023
1output_conv/batch_normalization_11/ReadVariableOpг
3output_conv/batch_normalization_11/ReadVariableOp_1ReadVariableOp<output_conv_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:*
dtype025
3output_conv/batch_normalization_11/ReadVariableOp_1Р
Boutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpKoutput_conv_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Boutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpЦ
Doutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMoutput_conv_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Doutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1 
3output_conv/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3&output_conv/conv2d_11/BiasAdd:output:09output_conv/batch_normalization_11/ReadVariableOp:value:0;output_conv/batch_normalization_11/ReadVariableOp_1:value:0Joutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Loutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<25
3output_conv/batch_normalization_11/FusedBatchNormV3€
1output_conv/batch_normalization_11/AssignNewValueAssignVariableOpKoutput_conv_batch_normalization_11_fusedbatchnormv3_readvariableop_resource@output_conv/batch_normalization_11/FusedBatchNormV3:batch_mean:0C^output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*^
_classT
RPloc:@output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1output_conv/batch_normalization_11/AssignNewValueН
3output_conv/batch_normalization_11/AssignNewValue_1AssignVariableOpMoutput_conv_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resourceDoutput_conv/batch_normalization_11/FusedBatchNormV3:batch_variance:0E^output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*`
_classV
TRloc:@output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3output_conv/batch_normalization_11/AssignNewValue_1°
output_conv/reshape_1/ShapeShape7output_conv/batch_normalization_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
output_conv/reshape_1/Shape†
)output_conv/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)output_conv/reshape_1/strided_slice/stack§
+output_conv/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+output_conv/reshape_1/strided_slice/stack_1§
+output_conv/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+output_conv/reshape_1/strided_slice/stack_2ж
#output_conv/reshape_1/strided_sliceStridedSlice$output_conv/reshape_1/Shape:output:02output_conv/reshape_1/strided_slice/stack:output:04output_conv/reshape_1/strided_slice/stack_1:output:04output_conv/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#output_conv/reshape_1/strided_sliceЩ
%output_conv/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2'
%output_conv/reshape_1/Reshape/shape/1ё
#output_conv/reshape_1/Reshape/shapePack,output_conv/reshape_1/strided_slice:output:0.output_conv/reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#output_conv/reshape_1/Reshape/shapeв
output_conv/reshape_1/ReshapeReshape7output_conv/batch_normalization_11/FusedBatchNormV3:y:0,output_conv/reshape_1/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
output_conv/reshape_1/ReshapeЗ
softmax/SoftmaxSoftmax&output_conv/reshape_1/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
softmax/Softmaxѓ
IdentityIdentitysoftmax/Softmax:softmax:0>^feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp@^feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1-^feature/batch_normalization_8/ReadVariableOp/^feature/batch_normalization_8/ReadVariableOp_1(^feature/conv2d_8/BiasAdd/ReadVariableOp'^feature/conv2d_8/Conv2D/ReadVariableOp?^input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpA^input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^input_conv/batch_normalization/ReadVariableOp0^input_conv/batch_normalization/ReadVariableOp_1A^input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^input_conv/batch_normalization_1/ReadVariableOp2^input_conv/batch_normalization_1/ReadVariableOp_1)^input_conv/conv2d/BiasAdd/ReadVariableOp(^input_conv/conv2d/Conv2D/ReadVariableOp+^input_conv/conv2d_1/BiasAdd/ReadVariableOp*^input_conv/conv2d_1/Conv2D/ReadVariableOp2^output_conv/batch_normalization_11/AssignNewValue4^output_conv/batch_normalization_11/AssignNewValue_1C^output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpE^output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12^output_conv/batch_normalization_11/ReadVariableOp4^output_conv/batch_normalization_11/ReadVariableOp_1-^output_conv/conv2d_11/BiasAdd/ReadVariableOp,^output_conv/conv2d_11/Conv2D/ReadVariableOpB^res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_2/ReadVariableOp3^res_block_0/batch_normalization_2/ReadVariableOp_1B^res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_3/ReadVariableOp3^res_block_0/batch_normalization_3/ReadVariableOp_1B^res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_4/ReadVariableOp3^res_block_0/batch_normalization_4/ReadVariableOp_1,^res_block_0/conv2d_2/BiasAdd/ReadVariableOp+^res_block_0/conv2d_2/Conv2D/ReadVariableOp,^res_block_0/conv2d_3/BiasAdd/ReadVariableOp+^res_block_0/conv2d_3/Conv2D/ReadVariableOp,^res_block_0/conv2d_4/BiasAdd/ReadVariableOp+^res_block_0/conv2d_4/Conv2D/ReadVariableOpB^res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_5/ReadVariableOp3^res_block_1/batch_normalization_5/ReadVariableOp_1B^res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_6/ReadVariableOp3^res_block_1/batch_normalization_6/ReadVariableOp_1B^res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_7/ReadVariableOp3^res_block_1/batch_normalization_7/ReadVariableOp_1,^res_block_1/conv2d_5/BiasAdd/ReadVariableOp+^res_block_1/conv2d_5/Conv2D/ReadVariableOp,^res_block_1/conv2d_6/BiasAdd/ReadVariableOp+^res_block_1/conv2d_6/Conv2D/ReadVariableOp,^res_block_1/conv2d_7/BiasAdd/ReadVariableOp+^res_block_1/conv2d_7/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*°
_input_shapesП
М:€€€€€€€€€ь::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2~
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2В
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12\
,feature/batch_normalization_8/ReadVariableOp,feature/batch_normalization_8/ReadVariableOp2`
.feature/batch_normalization_8/ReadVariableOp_1.feature/batch_normalization_8/ReadVariableOp_12R
'feature/conv2d_8/BiasAdd/ReadVariableOp'feature/conv2d_8/BiasAdd/ReadVariableOp2P
&feature/conv2d_8/Conv2D/ReadVariableOp&feature/conv2d_8/Conv2D/ReadVariableOp2А
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp2Д
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-input_conv/batch_normalization/ReadVariableOp-input_conv/batch_normalization/ReadVariableOp2b
/input_conv/batch_normalization/ReadVariableOp_1/input_conv/batch_normalization/ReadVariableOp_12Д
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2И
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12b
/input_conv/batch_normalization_1/ReadVariableOp/input_conv/batch_normalization_1/ReadVariableOp2f
1input_conv/batch_normalization_1/ReadVariableOp_11input_conv/batch_normalization_1/ReadVariableOp_12T
(input_conv/conv2d/BiasAdd/ReadVariableOp(input_conv/conv2d/BiasAdd/ReadVariableOp2R
'input_conv/conv2d/Conv2D/ReadVariableOp'input_conv/conv2d/Conv2D/ReadVariableOp2X
*input_conv/conv2d_1/BiasAdd/ReadVariableOp*input_conv/conv2d_1/BiasAdd/ReadVariableOp2V
)input_conv/conv2d_1/Conv2D/ReadVariableOp)input_conv/conv2d_1/Conv2D/ReadVariableOp2f
1output_conv/batch_normalization_11/AssignNewValue1output_conv/batch_normalization_11/AssignNewValue2j
3output_conv/batch_normalization_11/AssignNewValue_13output_conv/batch_normalization_11/AssignNewValue_12И
Boutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpBoutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2М
Doutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Doutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12f
1output_conv/batch_normalization_11/ReadVariableOp1output_conv/batch_normalization_11/ReadVariableOp2j
3output_conv/batch_normalization_11/ReadVariableOp_13output_conv/batch_normalization_11/ReadVariableOp_12\
,output_conv/conv2d_11/BiasAdd/ReadVariableOp,output_conv/conv2d_11/BiasAdd/ReadVariableOp2Z
+output_conv/conv2d_11/Conv2D/ReadVariableOp+output_conv/conv2d_11/Conv2D/ReadVariableOp2Ж
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2К
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_2/ReadVariableOp0res_block_0/batch_normalization_2/ReadVariableOp2h
2res_block_0/batch_normalization_2/ReadVariableOp_12res_block_0/batch_normalization_2/ReadVariableOp_12Ж
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2К
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_3/ReadVariableOp0res_block_0/batch_normalization_3/ReadVariableOp2h
2res_block_0/batch_normalization_3/ReadVariableOp_12res_block_0/batch_normalization_3/ReadVariableOp_12Ж
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2К
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_4/ReadVariableOp0res_block_0/batch_normalization_4/ReadVariableOp2h
2res_block_0/batch_normalization_4/ReadVariableOp_12res_block_0/batch_normalization_4/ReadVariableOp_12Z
+res_block_0/conv2d_2/BiasAdd/ReadVariableOp+res_block_0/conv2d_2/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_2/Conv2D/ReadVariableOp*res_block_0/conv2d_2/Conv2D/ReadVariableOp2Z
+res_block_0/conv2d_3/BiasAdd/ReadVariableOp+res_block_0/conv2d_3/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_3/Conv2D/ReadVariableOp*res_block_0/conv2d_3/Conv2D/ReadVariableOp2Z
+res_block_0/conv2d_4/BiasAdd/ReadVariableOp+res_block_0/conv2d_4/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_4/Conv2D/ReadVariableOp*res_block_0/conv2d_4/Conv2D/ReadVariableOp2Ж
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2К
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_5/ReadVariableOp0res_block_1/batch_normalization_5/ReadVariableOp2h
2res_block_1/batch_normalization_5/ReadVariableOp_12res_block_1/batch_normalization_5/ReadVariableOp_12Ж
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2К
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_6/ReadVariableOp0res_block_1/batch_normalization_6/ReadVariableOp2h
2res_block_1/batch_normalization_6/ReadVariableOp_12res_block_1/batch_normalization_6/ReadVariableOp_12Ж
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2К
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
:€€€€€€€€€ь
 
_user_specified_nameinputs
Ъ
ъ
.__inference_res_block_1_layer_call_fn_34130404

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
identityИҐStatefulPartitionedCallЁ
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
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_341271612
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34130896

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
м	
я
F__inference_conv2d_5_layer_call_and_return_conditional_losses_34126665

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
П"
€
H__inference_input_conv_layer_call_and_return_conditional_losses_34125341
conv2d_input
conv2d_34125141
conv2d_34125143 
batch_normalization_34125208 
batch_normalization_34125210 
batch_normalization_34125212 
batch_normalization_34125214
conv2d_1_34125251
conv2d_1_34125253"
batch_normalization_1_34125318"
batch_normalization_1_34125320"
batch_normalization_1_34125322"
batch_normalization_1_34125324
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCall•
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_34125141conv2d_34125143*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_341251302 
conv2d/StatefulPartitionedCallЅ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_34125208batch_normalization_34125210batch_normalization_34125212batch_normalization_34125214*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_341251632-
+batch_normalization/StatefulPartitionedCallЩ
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_341252222
activation/PartitionedCallƒ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_34125251conv2d_1_34125253*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_341252402"
 conv2d_1/StatefulPartitionedCallѕ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_34125318batch_normalization_1_34125320batch_normalization_1_34125322batch_normalization_1_34125324*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_341252732/
-batch_normalization_1/StatefulPartitionedCallЯ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_341253322
activation_1/PartitionedCall£
IdentityIdentity%activation_1/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€ь::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:^ Z
0
_output_shapes
:€€€€€€€€€ь
&
_user_specified_nameconv2d_input
Ё
g
K__inference_output_linear_layer_call_and_return_conditional_losses_34130611

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ќ
ф
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34124974

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3н
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
а
Ђ
8__inference_batch_normalization_4_layer_call_fn_34131368

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_341260692
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34126698

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ј
E
)__inference_relu_1_layer_call_fn_34130414

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_relu_1_layer_call_and_return_conditional_losses_341281722
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34131633

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
м	
я
F__inference_conv2d_6_layer_call_and_return_conditional_losses_34126775

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34126808

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ц
h
L__inference_feature_linear_layer_call_and_return_conditional_losses_34128233

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
ќ2
Ш
I__inference_output_conv_layer_call_and_return_conditional_losses_34130541

inputs,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource2
.batch_normalization_11_readvariableop_resource4
0batch_normalization_11_readvariableop_1_resourceC
?batch_normalization_11_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource
identityИҐ%batch_normalization_11/AssignNewValueҐ'batch_normalization_11/AssignNewValue_1Ґ6batch_normalization_11/FusedBatchNormV3/ReadVariableOpҐ8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Ґ%batch_normalization_11/ReadVariableOpҐ'batch_normalization_11/ReadVariableOp_1Ґ conv2d_11/BiasAdd/ReadVariableOpҐconv2d_11/Conv2D/ReadVariableOp≥
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02!
conv2d_11/Conv2D/ReadVariableOp¬
conv2d_11/Conv2DConv2Dinputs'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv2d_11/Conv2D™
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp∞
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_11/BiasAddє
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_11/ReadVariableOpњ
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_11/ReadVariableOp_1м
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpт
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ц
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_11/BiasAdd:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2)
'batch_normalization_11/FusedBatchNormV3Ј
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_11/AssignNewValue≈
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_11/AssignNewValue_1}
reshape_1/ShapeShape+batch_normalization_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
reshape_1/ShapeИ
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stackМ
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1М
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2Ю
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_sliceБ
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
reshape_1/Reshape/shape/1Ѓ
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape≤
reshape_1/ReshapeReshape+batch_normalization_11/FusedBatchNormV3:y:0 reshape_1/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
reshape_1/ReshapeЋ
IdentityIdentityreshape_1/Reshape:output:0&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R::::::2N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
Й
ф
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34130743

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1–
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:€€€€€€€€€®А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3№
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:€€€€€€€€€®А2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:€€€€€€€€€®А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:€€€€€€€€€®А
 
_user_specified_nameinputs
•
ґ
E__inference_feature_layer_call_and_return_conditional_losses_34130464

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource
identityИҐ5batch_normalization_8/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_8/ReadVariableOpҐ&batch_normalization_8/ReadVariableOp_1Ґconv2d_8/BiasAdd/ReadVariableOpҐconv2d_8/Conv2D/ReadVariableOp∞
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_8/Conv2D/ReadVariableOpњ
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R*
paddingVALID*
strides
2
conv2d_8/Conv2DІ
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOpђ
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R2
conv2d_8/BiasAddґ
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOpЉ
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1й
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3Ф
activation_6/ReluRelu*batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R2
activation_6/ReluА
IdentityIdentityactivation_6/Relu:activations:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1 ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R@::::::2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ё
d
H__inference_activation_layer_call_and_return_conditional_losses_34130774

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:€€€€€€€€€®А2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€®А2

Identity"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€®А:Y U
1
_output_shapes
:€€€€€€€€€®А
 
_user_specified_nameinputs
м	
я
F__inference_conv2d_3_layer_call_and_return_conditional_losses_34131095

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
м	
я
F__inference_conv2d_3_layer_call_and_return_conditional_losses_34125926

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
®
Ђ
8__inference_batch_normalization_3_layer_call_fn_34131166

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_341256912
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ь
√
*__inference_feature_layer_call_fn_34127507
conv2d_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_341274922
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R@::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€R@
(
_user_specified_nameconv2d_8_input
ы
ц
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34131867

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
®
Ђ
8__inference_batch_normalization_8_layer_call_fn_34131960

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_341272582
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ж
А
+__inference_conv2d_8_layer_call_fn_34131849

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_8_layer_call_and_return_conditional_losses_341273142
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
о	
а
G__inference_conv2d_11_layer_call_and_return_conditional_losses_34127625

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
≤
В
.__inference_res_block_1_layer_call_fn_34127200
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
identityИҐStatefulPartitionedCallе
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
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_341271612
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€R@
(
_user_specified_nameconv2d_5_input
√
ц
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34126640

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ƒ
ч
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34132104

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ь
њ
.__inference_output_conv_layer_call_fn_34130607

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCall≤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_output_conv_layer_call_and_return_conditional_losses_341278122
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
с

c
G__inference_reshape_1_layer_call_and_return_conditional_losses_34132142

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
strided_slice/stack_2в
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
€€€€€€€€€2
Reshape/shape/1Ж
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
т	
я
F__inference_conv2d_1_layer_call_and_return_conditional_losses_34125240

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€®А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€®А
 
_user_specified_nameinputs
т	
я
F__inference_conv2d_1_layer_call_and_return_conditional_losses_34130789

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:€€€€€€€€€®А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€®А
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34131355

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ј
E
)__inference_relu_0_layer_call_fn_34130190

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_relu_0_layer_call_and_return_conditional_losses_341280392
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
к:
у	
H__inference_input_conv_layer_call_and_return_conditional_losses_34129862

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
identityИҐ3batch_normalization/FusedBatchNormV3/ReadVariableOpҐ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ"batch_normalization/ReadVariableOpҐ$batch_normalization/ReadVariableOp_1Ґ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_1/ReadVariableOpҐ&batch_normalization_1/ReadVariableOp_1Ґconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpЂ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02
conv2d/Conv2D/ReadVariableOpї
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€®А*
paddingVALID*
strides
2
conv2d/Conv2DҐ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
conv2d/BiasAdd/ReadVariableOp¶
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€®А2
conv2d/BiasAdd±
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"batch_normalization/ReadVariableOpЈ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:А*
dtype02&
$batch_normalization/ReadVariableOp_1д
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpк
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ў
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:€€€€€€€€€®А:А:А:А:А:*
epsilon%oГ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3Р
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:€€€€€€€€€®А2
activation/Relu±
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp÷
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingVALID*
strides
2
conv2d_1/Conv2DІ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
conv2d_1/BiasAddґ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOpЉ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1й
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3Ф
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
activation_1/Reluщ
IdentityIdentityactivation_1/Relu:activations:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€ь::::::::::::2j
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
:€€€€€€€€€ь
 
_user_specified_nameinputs
Ь/
£
I__inference_res_block_0_layer_call_and_return_conditional_losses_34126124
conv2d_2_input
conv2d_2_34125827
conv2d_2_34125829"
batch_normalization_2_34125894"
batch_normalization_2_34125896"
batch_normalization_2_34125898"
batch_normalization_2_34125900
conv2d_3_34125937
conv2d_3_34125939"
batch_normalization_3_34126004"
batch_normalization_3_34126006"
batch_normalization_3_34126008"
batch_normalization_3_34126010
conv2d_4_34126047
conv2d_4_34126049"
batch_normalization_4_34126114"
batch_normalization_4_34126116"
batch_normalization_4_34126118"
batch_normalization_4_34126120
identityИҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallѓ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_34125827conv2d_2_34125829*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_341258162"
 conv2d_2/StatefulPartitionedCallѕ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_34125894batch_normalization_2_34125896batch_normalization_2_34125898batch_normalization_2_34125900*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_341258492/
-batch_normalization_2/StatefulPartitionedCallЯ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_341259082
activation_2/PartitionedCall∆
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_34125937conv2d_3_34125939*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_341259262"
 conv2d_3/StatefulPartitionedCallѕ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_34126004batch_normalization_3_34126006batch_normalization_3_34126008batch_normalization_3_34126010*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_341259592/
-batch_normalization_3/StatefulPartitionedCallЯ
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_3_layer_call_and_return_conditional_losses_341260182
activation_3/PartitionedCall∆
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_34126047conv2d_4_34126049*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_341260362"
 conv2d_4/StatefulPartitionedCallѕ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_34126114batch_normalization_4_34126116batch_normalization_4_34126118batch_normalization_4_34126120*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_341260692/
-batch_normalization_4/StatefulPartitionedCallЛ
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€R@
(
_user_specified_nameconv2d_2_input
а
Ђ
8__inference_batch_normalization_2_layer_call_fn_34131013

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_341258672
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
м	
я
F__inference_conv2d_7_layer_call_and_return_conditional_losses_34126885

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
®
Ђ
8__inference_batch_normalization_1_layer_call_fn_34130860

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_341251052
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ё
d
H__inference_activation_layer_call_and_return_conditional_losses_34125222

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:€€€€€€€€€®А2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€®А2

Identity"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€®А:Y U
1
_output_shapes
:€€€€€€€€€®А
 
_user_specified_nameinputs
ьV
ё
I__inference_res_block_0_layer_call_and_return_conditional_losses_34130032

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
identityИҐ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґ5batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_3/ReadVariableOpҐ&batch_normalization_3/ReadVariableOp_1Ґ5batch_normalization_4/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_4/ReadVariableOpҐ&batch_normalization_4/ReadVariableOp_1Ґconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOpҐconv2d_3/BiasAdd/ReadVariableOpҐconv2d_3/Conv2D/ReadVariableOpҐconv2d_4/BiasAdd/ReadVariableOpҐconv2d_4/Conv2D/ReadVariableOp∞
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpЊ
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
conv2d_2/Conv2DІ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpђ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
conv2d_2/BiasAddґ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOpЉ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1й
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3Ф
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
activation_2/Relu∞
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp„
conv2d_3/Conv2DConv2Dactivation_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
conv2d_3/Conv2DІ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpђ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
conv2d_3/BiasAddґ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOpЉ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1й
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3Ф
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
activation_3/Relu∞
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp„
conv2d_4/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
conv2d_4/Conv2DІ
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOpђ
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
conv2d_4/BiasAddґ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOpЉ
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1й
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3Х
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::2n
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
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34125760

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
√
K
/__inference_activation_5_layer_call_fn_34131687

inputs
identity’
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_5_layer_call_and_return_conditional_losses_341268672
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34131031

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34131742

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Д/
Ы
I__inference_res_block_0_layer_call_and_return_conditional_losses_34126312

inputs
conv2d_2_34126267
conv2d_2_34126269"
batch_normalization_2_34126272"
batch_normalization_2_34126274"
batch_normalization_2_34126276"
batch_normalization_2_34126278
conv2d_3_34126282
conv2d_3_34126284"
batch_normalization_3_34126287"
batch_normalization_3_34126289"
batch_normalization_3_34126291"
batch_normalization_3_34126293
conv2d_4_34126297
conv2d_4_34126299"
batch_normalization_4_34126302"
batch_normalization_4_34126304"
batch_normalization_4_34126306"
batch_normalization_4_34126308
identityИҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallІ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_34126267conv2d_2_34126269*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_341258162"
 conv2d_2/StatefulPartitionedCallѕ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_34126272batch_normalization_2_34126274batch_normalization_2_34126276batch_normalization_2_34126278*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_341258672/
-batch_normalization_2/StatefulPartitionedCallЯ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_341259082
activation_2/PartitionedCall∆
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_34126282conv2d_3_34126284*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_341259262"
 conv2d_3/StatefulPartitionedCallѕ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_34126287batch_normalization_3_34126289batch_normalization_3_34126291batch_normalization_3_34126293*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_341259772/
-batch_normalization_3/StatefulPartitionedCallЯ
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_3_layer_call_and_return_conditional_losses_341260182
activation_3/PartitionedCall∆
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_34126297conv2d_4_34126299*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_341260362"
 conv2d_4/StatefulPartitionedCallѕ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_34126302batch_normalization_4_34126304batch_normalization_4_34126306batch_normalization_4_34126308*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_341260872/
-batch_normalization_4/StatefulPartitionedCallЛ
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ќ	
Щ
-__inference_input_conv_layer_call_fn_34129966

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
identityИҐStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_341254752
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€ь::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ь
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34126440

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34126540

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
а
Ђ
8__inference_batch_normalization_5_layer_call_fn_34131511

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_341266982
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
®
Ђ
8__inference_batch_normalization_8_layer_call_fn_34131973

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_341272892
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ж
А
+__inference_conv2d_3_layer_call_fn_34131104

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_341259262
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34126087

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Д/
Ы
I__inference_res_block_1_layer_call_and_return_conditional_losses_34127161

inputs
conv2d_5_34127116
conv2d_5_34127118"
batch_normalization_5_34127121"
batch_normalization_5_34127123"
batch_normalization_5_34127125"
batch_normalization_5_34127127
conv2d_6_34127131
conv2d_6_34127133"
batch_normalization_6_34127136"
batch_normalization_6_34127138"
batch_normalization_6_34127140"
batch_normalization_6_34127142
conv2d_7_34127146
conv2d_7_34127148"
batch_normalization_7_34127151"
batch_normalization_7_34127153"
batch_normalization_7_34127155"
batch_normalization_7_34127157
identityИҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐ-batch_normalization_7/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallІ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_34127116conv2d_5_34127118*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_341266652"
 conv2d_5/StatefulPartitionedCallѕ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_34127121batch_normalization_5_34127123batch_normalization_5_34127125batch_normalization_5_34127127*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_341267162/
-batch_normalization_5/StatefulPartitionedCallЯ
activation_4/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_4_layer_call_and_return_conditional_losses_341267572
activation_4/PartitionedCall∆
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_34127131conv2d_6_34127133*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_341267752"
 conv2d_6/StatefulPartitionedCallѕ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_34127136batch_normalization_6_34127138batch_normalization_6_34127140batch_normalization_6_34127142*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_341268262/
-batch_normalization_6/StatefulPartitionedCallЯ
activation_5/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_5_layer_call_and_return_conditional_losses_341268672
activation_5/PartitionedCall∆
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_34127146conv2d_7_34127148*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_7_layer_call_and_return_conditional_losses_341268852"
 conv2d_7/StatefulPartitionedCallѕ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_34127151batch_normalization_7_34127153batch_normalization_7_34127155batch_normalization_7_34127157*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_341269362/
-batch_normalization_7/StatefulPartitionedCallЛ
IdentityIdentity6batch_normalization_7/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
З
~
)__inference_conv2d_layer_call_fn_34130645

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_341251302
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€®А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€ь::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ь
 
_user_specified_nameinputs
а
Ђ
8__inference_batch_normalization_5_layer_call_fn_34131524

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_341267162
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
а
Ђ
8__inference_batch_normalization_7_layer_call_fn_34131768

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_341269362
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34131436

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
як
д9
E__inference_model_1_layer_call_and_return_conditional_losses_34129566

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
Hfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource8
4output_conv_conv2d_11_conv2d_readvariableop_resource9
5output_conv_conv2d_11_biasadd_readvariableop_resource>
:output_conv_batch_normalization_11_readvariableop_resource@
<output_conv_batch_normalization_11_readvariableop_1_resourceO
Koutput_conv_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceQ
Moutput_conv_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource
identityИҐ=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpҐ?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Ґ,feature/batch_normalization_8/ReadVariableOpҐ.feature/batch_normalization_8/ReadVariableOp_1Ґ'feature/conv2d_8/BiasAdd/ReadVariableOpҐ&feature/conv2d_8/Conv2D/ReadVariableOpҐ>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpҐ@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ-input_conv/batch_normalization/ReadVariableOpҐ/input_conv/batch_normalization/ReadVariableOp_1Ґ@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐBinput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ/input_conv/batch_normalization_1/ReadVariableOpҐ1input_conv/batch_normalization_1/ReadVariableOp_1Ґ(input_conv/conv2d/BiasAdd/ReadVariableOpҐ'input_conv/conv2d/Conv2D/ReadVariableOpҐ*input_conv/conv2d_1/BiasAdd/ReadVariableOpҐ)input_conv/conv2d_1/Conv2D/ReadVariableOpҐBoutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpҐDoutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Ґ1output_conv/batch_normalization_11/ReadVariableOpҐ3output_conv/batch_normalization_11/ReadVariableOp_1Ґ,output_conv/conv2d_11/BiasAdd/ReadVariableOpҐ+output_conv/conv2d_11/Conv2D/ReadVariableOpҐAres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐCres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ0res_block_0/batch_normalization_2/ReadVariableOpҐ2res_block_0/batch_normalization_2/ReadVariableOp_1ҐAres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐCres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ0res_block_0/batch_normalization_3/ReadVariableOpҐ2res_block_0/batch_normalization_3/ReadVariableOp_1ҐAres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpҐCres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ґ0res_block_0/batch_normalization_4/ReadVariableOpҐ2res_block_0/batch_normalization_4/ReadVariableOp_1Ґ+res_block_0/conv2d_2/BiasAdd/ReadVariableOpҐ*res_block_0/conv2d_2/Conv2D/ReadVariableOpҐ+res_block_0/conv2d_3/BiasAdd/ReadVariableOpҐ*res_block_0/conv2d_3/Conv2D/ReadVariableOpҐ+res_block_0/conv2d_4/BiasAdd/ReadVariableOpҐ*res_block_0/conv2d_4/Conv2D/ReadVariableOpҐAres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpҐCres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ґ0res_block_1/batch_normalization_5/ReadVariableOpҐ2res_block_1/batch_normalization_5/ReadVariableOp_1ҐAres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpҐCres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ґ0res_block_1/batch_normalization_6/ReadVariableOpҐ2res_block_1/batch_normalization_6/ReadVariableOp_1ҐAres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpҐCres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ґ0res_block_1/batch_normalization_7/ReadVariableOpҐ2res_block_1/batch_normalization_7/ReadVariableOp_1Ґ+res_block_1/conv2d_5/BiasAdd/ReadVariableOpҐ*res_block_1/conv2d_5/Conv2D/ReadVariableOpҐ+res_block_1/conv2d_6/BiasAdd/ReadVariableOpҐ*res_block_1/conv2d_6/Conv2D/ReadVariableOpҐ+res_block_1/conv2d_7/BiasAdd/ReadVariableOpҐ*res_block_1/conv2d_7/Conv2D/ReadVariableOpћ
'input_conv/conv2d/Conv2D/ReadVariableOpReadVariableOp0input_conv_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02)
'input_conv/conv2d/Conv2D/ReadVariableOp№
input_conv/conv2d/Conv2DConv2Dinputs/input_conv/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€®А*
paddingVALID*
strides
2
input_conv/conv2d/Conv2D√
(input_conv/conv2d/BiasAdd/ReadVariableOpReadVariableOp1input_conv_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(input_conv/conv2d/BiasAdd/ReadVariableOp“
input_conv/conv2d/BiasAddBiasAdd!input_conv/conv2d/Conv2D:output:00input_conv/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€®А2
input_conv/conv2d/BiasAdd“
-input_conv/batch_normalization/ReadVariableOpReadVariableOp6input_conv_batch_normalization_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-input_conv/batch_normalization/ReadVariableOpЎ
/input_conv/batch_normalization/ReadVariableOp_1ReadVariableOp8input_conv_batch_normalization_readvariableop_1_resource*
_output_shapes	
:А*
dtype021
/input_conv/batch_normalization/ReadVariableOp_1Е
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02@
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpЛ
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIinput_conv_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02B
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¶
/input_conv/batch_normalization/FusedBatchNormV3FusedBatchNormV3"input_conv/conv2d/BiasAdd:output:05input_conv/batch_normalization/ReadVariableOp:value:07input_conv/batch_normalization/ReadVariableOp_1:value:0Finput_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hinput_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:€€€€€€€€€®А:А:А:А:А:*
epsilon%oГ:*
is_training( 21
/input_conv/batch_normalization/FusedBatchNormV3±
input_conv/activation/ReluRelu3input_conv/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:€€€€€€€€€®А2
input_conv/activation/Relu“
)input_conv/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2input_conv_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype02+
)input_conv/conv2d_1/Conv2D/ReadVariableOpВ
input_conv/conv2d_1/Conv2DConv2D(input_conv/activation/Relu:activations:01input_conv/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingVALID*
strides
2
input_conv/conv2d_1/Conv2D»
*input_conv/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3input_conv_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*input_conv/conv2d_1/BiasAdd/ReadVariableOpЎ
input_conv/conv2d_1/BiasAddBiasAdd#input_conv/conv2d_1/Conv2D:output:02input_conv/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
input_conv/conv2d_1/BiasAdd„
/input_conv/batch_normalization_1/ReadVariableOpReadVariableOp8input_conv_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype021
/input_conv/batch_normalization_1/ReadVariableOpЁ
1input_conv/batch_normalization_1/ReadVariableOp_1ReadVariableOp:input_conv_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1input_conv/batch_normalization_1/ReadVariableOp_1К
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02B
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpР
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKinput_conv_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ѓ
1input_conv/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$input_conv/conv2d_1/BiasAdd:output:07input_conv/batch_normalization_1/ReadVariableOp:value:09input_conv/batch_normalization_1/ReadVariableOp_1:value:0Hinput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jinput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 23
1input_conv/batch_normalization_1/FusedBatchNormV3µ
input_conv/activation_1/ReluRelu5input_conv/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
input_conv/activation_1/Relu‘
*res_block_0/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_2/Conv2D/ReadVariableOpЖ
res_block_0/conv2d_2/Conv2DConv2D*input_conv/activation_1/Relu:activations:02res_block_0/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
res_block_0/conv2d_2/Conv2DЋ
+res_block_0/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_2/BiasAdd/ReadVariableOp№
res_block_0/conv2d_2/BiasAddBiasAdd$res_block_0/conv2d_2/Conv2D:output:03res_block_0/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_0/conv2d_2/BiasAddЏ
0res_block_0/batch_normalization_2/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_2/ReadVariableOpа
2res_block_0/batch_normalization_2/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_2/ReadVariableOp_1Н
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpУ
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1µ
2res_block_0/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_2/BiasAdd:output:08res_block_0/batch_normalization_2/ReadVariableOp:value:0:res_block_0/batch_normalization_2/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 24
2res_block_0/batch_normalization_2/FusedBatchNormV3Є
res_block_0/activation_2/ReluRelu6res_block_0/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_0/activation_2/Relu‘
*res_block_0/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_3/Conv2D/ReadVariableOpЗ
res_block_0/conv2d_3/Conv2DConv2D+res_block_0/activation_2/Relu:activations:02res_block_0/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
res_block_0/conv2d_3/Conv2DЋ
+res_block_0/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_3/BiasAdd/ReadVariableOp№
res_block_0/conv2d_3/BiasAddBiasAdd$res_block_0/conv2d_3/Conv2D:output:03res_block_0/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_0/conv2d_3/BiasAddЏ
0res_block_0/batch_normalization_3/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_3/ReadVariableOpа
2res_block_0/batch_normalization_3/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_3/ReadVariableOp_1Н
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpУ
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1µ
2res_block_0/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_3/BiasAdd:output:08res_block_0/batch_normalization_3/ReadVariableOp:value:0:res_block_0/batch_normalization_3/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 24
2res_block_0/batch_normalization_3/FusedBatchNormV3Є
res_block_0/activation_3/ReluRelu6res_block_0/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_0/activation_3/Relu‘
*res_block_0/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3res_block_0_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_0/conv2d_4/Conv2D/ReadVariableOpЗ
res_block_0/conv2d_4/Conv2DConv2D+res_block_0/activation_3/Relu:activations:02res_block_0/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
res_block_0/conv2d_4/Conv2DЋ
+res_block_0/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4res_block_0_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_0/conv2d_4/BiasAdd/ReadVariableOp№
res_block_0/conv2d_4/BiasAddBiasAdd$res_block_0/conv2d_4/Conv2D:output:03res_block_0/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_0/conv2d_4/BiasAddЏ
0res_block_0/batch_normalization_4/ReadVariableOpReadVariableOp9res_block_0_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_0/batch_normalization_4/ReadVariableOpа
2res_block_0/batch_normalization_4/ReadVariableOp_1ReadVariableOp;res_block_0_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_0/batch_normalization_4/ReadVariableOp_1Н
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpУ
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_0_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1µ
2res_block_0/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%res_block_0/conv2d_4/BiasAdd:output:08res_block_0/batch_normalization_4/ReadVariableOp:value:0:res_block_0/batch_normalization_4/ReadVariableOp_1:value:0Ires_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 24
2res_block_0/batch_normalization_4/FusedBatchNormV3я
tf.__operators__.add/AddV2AddV2*input_conv/activation_1/Relu:activations:06res_block_0/batch_normalization_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
tf.__operators__.add/AddV2|
relu_0/ReluRelutf.__operators__.add/AddV2:z:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
relu_0/Relu‘
*res_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_5/Conv2D/ReadVariableOpх
res_block_1/conv2d_5/Conv2DConv2Drelu_0/Relu:activations:02res_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
res_block_1/conv2d_5/Conv2DЋ
+res_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_5/BiasAdd/ReadVariableOp№
res_block_1/conv2d_5/BiasAddBiasAdd$res_block_1/conv2d_5/Conv2D:output:03res_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_1/conv2d_5/BiasAddЏ
0res_block_1/batch_normalization_5/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_5/ReadVariableOpа
2res_block_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_5/ReadVariableOp_1Н
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpУ
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1µ
2res_block_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_5/BiasAdd:output:08res_block_1/batch_normalization_5/ReadVariableOp:value:0:res_block_1/batch_normalization_5/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 24
2res_block_1/batch_normalization_5/FusedBatchNormV3Є
res_block_1/activation_4/ReluRelu6res_block_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_1/activation_4/Relu‘
*res_block_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_6/Conv2D/ReadVariableOpЗ
res_block_1/conv2d_6/Conv2DConv2D+res_block_1/activation_4/Relu:activations:02res_block_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
res_block_1/conv2d_6/Conv2DЋ
+res_block_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_6/BiasAdd/ReadVariableOp№
res_block_1/conv2d_6/BiasAddBiasAdd$res_block_1/conv2d_6/Conv2D:output:03res_block_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_1/conv2d_6/BiasAddЏ
0res_block_1/batch_normalization_6/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_6/ReadVariableOpа
2res_block_1/batch_normalization_6/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_6/ReadVariableOp_1Н
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpУ
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1µ
2res_block_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_6/BiasAdd:output:08res_block_1/batch_normalization_6/ReadVariableOp:value:0:res_block_1/batch_normalization_6/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 24
2res_block_1/batch_normalization_6/FusedBatchNormV3Є
res_block_1/activation_5/ReluRelu6res_block_1/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_1/activation_5/Relu‘
*res_block_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3res_block_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*res_block_1/conv2d_7/Conv2D/ReadVariableOpЗ
res_block_1/conv2d_7/Conv2DConv2D+res_block_1/activation_5/Relu:activations:02res_block_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
res_block_1/conv2d_7/Conv2DЋ
+res_block_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4res_block_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+res_block_1/conv2d_7/BiasAdd/ReadVariableOp№
res_block_1/conv2d_7/BiasAddBiasAdd$res_block_1/conv2d_7/Conv2D:output:03res_block_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
res_block_1/conv2d_7/BiasAddЏ
0res_block_1/batch_normalization_7/ReadVariableOpReadVariableOp9res_block_1_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype022
0res_block_1/batch_normalization_7/ReadVariableOpа
2res_block_1/batch_normalization_7/ReadVariableOp_1ReadVariableOp;res_block_1_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2res_block_1/batch_normalization_7/ReadVariableOp_1Н
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpJres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpУ
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLres_block_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
Cres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1µ
2res_block_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%res_block_1/conv2d_7/BiasAdd:output:08res_block_1/batch_normalization_7/ReadVariableOp:value:0:res_block_1/batch_normalization_7/ReadVariableOp_1:value:0Ires_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Kres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 24
2res_block_1/batch_normalization_7/FusedBatchNormV3“
tf.__operators__.add_1/AddV2AddV2relu_0/Relu:activations:06res_block_1/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
tf.__operators__.add_1/AddV2~
relu_1/ReluRelu tf.__operators__.add_1/AddV2:z:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
relu_1/Relu»
&feature/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/feature_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&feature/conv2d_8/Conv2D/ReadVariableOpк
feature/conv2d_8/Conv2DConv2Drelu_1/Relu:activations:0.feature/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R*
paddingVALID*
strides
2
feature/conv2d_8/Conv2Dњ
'feature/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0feature_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'feature/conv2d_8/BiasAdd/ReadVariableOpћ
feature/conv2d_8/BiasAddBiasAdd feature/conv2d_8/Conv2D:output:0/feature/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R2
feature/conv2d_8/BiasAddќ
,feature/batch_normalization_8/ReadVariableOpReadVariableOp5feature_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02.
,feature/batch_normalization_8/ReadVariableOp‘
.feature/batch_normalization_8/ReadVariableOp_1ReadVariableOp7feature_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype020
.feature/batch_normalization_8/ReadVariableOp_1Б
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOpЗ
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHfeature_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Щ
.feature/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3!feature/conv2d_8/BiasAdd:output:04feature/batch_normalization_8/ReadVariableOp:value:06feature/batch_normalization_8/ReadVariableOp_1:value:0Efeature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gfeature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R:::::*
epsilon%oГ:*
is_training( 20
.feature/batch_normalization_8/FusedBatchNormV3ђ
feature/activation_6/ReluRelu2feature/batch_normalization_8/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R2
feature/activation_6/ReluІ
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/permд
 tf.compat.v1.transpose/transpose	Transpose'feature/activation_6/Relu:activations:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:€€€€€€€€€R2"
 tf.compat.v1.transpose/transpose„
+output_conv/conv2d_11/Conv2D/ReadVariableOpReadVariableOp4output_conv_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:R*
dtype02-
+output_conv/conv2d_11/Conv2D/ReadVariableOpД
output_conv/conv2d_11/Conv2DConv2D$tf.compat.v1.transpose/transpose:y:03output_conv/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
output_conv/conv2d_11/Conv2Dќ
,output_conv/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp5output_conv_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,output_conv/conv2d_11/BiasAdd/ReadVariableOpа
output_conv/conv2d_11/BiasAddBiasAdd%output_conv/conv2d_11/Conv2D:output:04output_conv/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
output_conv/conv2d_11/BiasAddЁ
1output_conv/batch_normalization_11/ReadVariableOpReadVariableOp:output_conv_batch_normalization_11_readvariableop_resource*
_output_shapes
:*
dtype023
1output_conv/batch_normalization_11/ReadVariableOpг
3output_conv/batch_normalization_11/ReadVariableOp_1ReadVariableOp<output_conv_batch_normalization_11_readvariableop_1_resource*
_output_shapes
:*
dtype025
3output_conv/batch_normalization_11/ReadVariableOp_1Р
Boutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpKoutput_conv_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Boutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpЦ
Doutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMoutput_conv_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Doutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Љ
3output_conv/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3&output_conv/conv2d_11/BiasAdd:output:09output_conv/batch_normalization_11/ReadVariableOp:value:0;output_conv/batch_normalization_11/ReadVariableOp_1:value:0Joutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Loutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 25
3output_conv/batch_normalization_11/FusedBatchNormV3°
output_conv/reshape_1/ShapeShape7output_conv/batch_normalization_11/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
output_conv/reshape_1/Shape†
)output_conv/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)output_conv/reshape_1/strided_slice/stack§
+output_conv/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+output_conv/reshape_1/strided_slice/stack_1§
+output_conv/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+output_conv/reshape_1/strided_slice/stack_2ж
#output_conv/reshape_1/strided_sliceStridedSlice$output_conv/reshape_1/Shape:output:02output_conv/reshape_1/strided_slice/stack:output:04output_conv/reshape_1/strided_slice/stack_1:output:04output_conv/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#output_conv/reshape_1/strided_sliceЩ
%output_conv/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2'
%output_conv/reshape_1/Reshape/shape/1ё
#output_conv/reshape_1/Reshape/shapePack,output_conv/reshape_1/strided_slice:output:0.output_conv/reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#output_conv/reshape_1/Reshape/shapeв
output_conv/reshape_1/ReshapeReshape7output_conv/batch_normalization_11/FusedBatchNormV3:y:0,output_conv/reshape_1/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
output_conv/reshape_1/ReshapeЗ
softmax/SoftmaxSoftmax&output_conv/reshape_1/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
softmax/Softmax≈
IdentityIdentitysoftmax/Softmax:softmax:0>^feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp@^feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1-^feature/batch_normalization_8/ReadVariableOp/^feature/batch_normalization_8/ReadVariableOp_1(^feature/conv2d_8/BiasAdd/ReadVariableOp'^feature/conv2d_8/Conv2D/ReadVariableOp?^input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOpA^input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^input_conv/batch_normalization/ReadVariableOp0^input_conv/batch_normalization/ReadVariableOp_1A^input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^input_conv/batch_normalization_1/ReadVariableOp2^input_conv/batch_normalization_1/ReadVariableOp_1)^input_conv/conv2d/BiasAdd/ReadVariableOp(^input_conv/conv2d/Conv2D/ReadVariableOp+^input_conv/conv2d_1/BiasAdd/ReadVariableOp*^input_conv/conv2d_1/Conv2D/ReadVariableOpC^output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpE^output_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12^output_conv/batch_normalization_11/ReadVariableOp4^output_conv/batch_normalization_11/ReadVariableOp_1-^output_conv/conv2d_11/BiasAdd/ReadVariableOp,^output_conv/conv2d_11/Conv2D/ReadVariableOpB^res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_2/ReadVariableOp3^res_block_0/batch_normalization_2/ReadVariableOp_1B^res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_3/ReadVariableOp3^res_block_0/batch_normalization_3/ReadVariableOp_1B^res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^res_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^res_block_0/batch_normalization_4/ReadVariableOp3^res_block_0/batch_normalization_4/ReadVariableOp_1,^res_block_0/conv2d_2/BiasAdd/ReadVariableOp+^res_block_0/conv2d_2/Conv2D/ReadVariableOp,^res_block_0/conv2d_3/BiasAdd/ReadVariableOp+^res_block_0/conv2d_3/Conv2D/ReadVariableOp,^res_block_0/conv2d_4/BiasAdd/ReadVariableOp+^res_block_0/conv2d_4/Conv2D/ReadVariableOpB^res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_5/ReadVariableOp3^res_block_1/batch_normalization_5/ReadVariableOp_1B^res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_6/ReadVariableOp3^res_block_1/batch_normalization_6/ReadVariableOp_1B^res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpD^res_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_11^res_block_1/batch_normalization_7/ReadVariableOp3^res_block_1/batch_normalization_7/ReadVariableOp_1,^res_block_1/conv2d_5/BiasAdd/ReadVariableOp+^res_block_1/conv2d_5/Conv2D/ReadVariableOp,^res_block_1/conv2d_6/BiasAdd/ReadVariableOp+^res_block_1/conv2d_6/Conv2D/ReadVariableOp,^res_block_1/conv2d_7/BiasAdd/ReadVariableOp+^res_block_1/conv2d_7/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*°
_input_shapesП
М:€€€€€€€€€ь::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2~
=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp=feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2В
?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?feature/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12\
,feature/batch_normalization_8/ReadVariableOp,feature/batch_normalization_8/ReadVariableOp2`
.feature/batch_normalization_8/ReadVariableOp_1.feature/batch_normalization_8/ReadVariableOp_12R
'feature/conv2d_8/BiasAdd/ReadVariableOp'feature/conv2d_8/BiasAdd/ReadVariableOp2P
&feature/conv2d_8/Conv2D/ReadVariableOp&feature/conv2d_8/Conv2D/ReadVariableOp2А
>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp>input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp2Д
@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@input_conv/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-input_conv/batch_normalization/ReadVariableOp-input_conv/batch_normalization/ReadVariableOp2b
/input_conv/batch_normalization/ReadVariableOp_1/input_conv/batch_normalization/ReadVariableOp_12Д
@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@input_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2И
Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Binput_conv/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12b
/input_conv/batch_normalization_1/ReadVariableOp/input_conv/batch_normalization_1/ReadVariableOp2f
1input_conv/batch_normalization_1/ReadVariableOp_11input_conv/batch_normalization_1/ReadVariableOp_12T
(input_conv/conv2d/BiasAdd/ReadVariableOp(input_conv/conv2d/BiasAdd/ReadVariableOp2R
'input_conv/conv2d/Conv2D/ReadVariableOp'input_conv/conv2d/Conv2D/ReadVariableOp2X
*input_conv/conv2d_1/BiasAdd/ReadVariableOp*input_conv/conv2d_1/BiasAdd/ReadVariableOp2V
)input_conv/conv2d_1/Conv2D/ReadVariableOp)input_conv/conv2d_1/Conv2D/ReadVariableOp2И
Boutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOpBoutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2М
Doutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Doutput_conv/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12f
1output_conv/batch_normalization_11/ReadVariableOp1output_conv/batch_normalization_11/ReadVariableOp2j
3output_conv/batch_normalization_11/ReadVariableOp_13output_conv/batch_normalization_11/ReadVariableOp_12\
,output_conv/conv2d_11/BiasAdd/ReadVariableOp,output_conv/conv2d_11/BiasAdd/ReadVariableOp2Z
+output_conv/conv2d_11/Conv2D/ReadVariableOp+output_conv/conv2d_11/Conv2D/ReadVariableOp2Ж
Ares_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2К
Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_2/ReadVariableOp0res_block_0/batch_normalization_2/ReadVariableOp2h
2res_block_0/batch_normalization_2/ReadVariableOp_12res_block_0/batch_normalization_2/ReadVariableOp_12Ж
Ares_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2К
Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_3/ReadVariableOp0res_block_0/batch_normalization_3/ReadVariableOp2h
2res_block_0/batch_normalization_3/ReadVariableOp_12res_block_0/batch_normalization_3/ReadVariableOp_12Ж
Ares_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2К
Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Cres_block_0/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0res_block_0/batch_normalization_4/ReadVariableOp0res_block_0/batch_normalization_4/ReadVariableOp2h
2res_block_0/batch_normalization_4/ReadVariableOp_12res_block_0/batch_normalization_4/ReadVariableOp_12Z
+res_block_0/conv2d_2/BiasAdd/ReadVariableOp+res_block_0/conv2d_2/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_2/Conv2D/ReadVariableOp*res_block_0/conv2d_2/Conv2D/ReadVariableOp2Z
+res_block_0/conv2d_3/BiasAdd/ReadVariableOp+res_block_0/conv2d_3/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_3/Conv2D/ReadVariableOp*res_block_0/conv2d_3/Conv2D/ReadVariableOp2Z
+res_block_0/conv2d_4/BiasAdd/ReadVariableOp+res_block_0/conv2d_4/BiasAdd/ReadVariableOp2X
*res_block_0/conv2d_4/Conv2D/ReadVariableOp*res_block_0/conv2d_4/Conv2D/ReadVariableOp2Ж
Ares_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2К
Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_5/ReadVariableOp0res_block_1/batch_normalization_5/ReadVariableOp2h
2res_block_1/batch_normalization_5/ReadVariableOp_12res_block_1/batch_normalization_5/ReadVariableOp_12Ж
Ares_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2К
Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Cres_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0res_block_1/batch_normalization_6/ReadVariableOp0res_block_1/batch_normalization_6/ReadVariableOp2h
2res_block_1/batch_normalization_6/ReadVariableOp_12res_block_1/batch_normalization_6/ReadVariableOp_12Ж
Ares_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpAres_block_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2К
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
:€€€€€€€€€ь
 
_user_specified_nameinputs
ђG
Р
E__inference_model_1_layer_call_and_return_conditional_losses_34128452	
input
input_conv_34128317
input_conv_34128319
input_conv_34128321
input_conv_34128323
input_conv_34128325
input_conv_34128327
input_conv_34128329
input_conv_34128331
input_conv_34128333
input_conv_34128335
input_conv_34128337
input_conv_34128339
res_block_0_34128342
res_block_0_34128344
res_block_0_34128346
res_block_0_34128348
res_block_0_34128350
res_block_0_34128352
res_block_0_34128354
res_block_0_34128356
res_block_0_34128358
res_block_0_34128360
res_block_0_34128362
res_block_0_34128364
res_block_0_34128366
res_block_0_34128368
res_block_0_34128370
res_block_0_34128372
res_block_0_34128374
res_block_0_34128376
res_block_1_34128381
res_block_1_34128383
res_block_1_34128385
res_block_1_34128387
res_block_1_34128389
res_block_1_34128391
res_block_1_34128393
res_block_1_34128395
res_block_1_34128397
res_block_1_34128399
res_block_1_34128401
res_block_1_34128403
res_block_1_34128405
res_block_1_34128407
res_block_1_34128409
res_block_1_34128411
res_block_1_34128413
res_block_1_34128415
feature_34128420
feature_34128422
feature_34128424
feature_34128426
feature_34128428
feature_34128430
output_conv_34128436
output_conv_34128438
output_conv_34128440
output_conv_34128442
output_conv_34128444
output_conv_34128446
identityИҐfeature/StatefulPartitionedCallҐ"input_conv/StatefulPartitionedCallҐ#output_conv/StatefulPartitionedCallҐ#res_block_0/StatefulPartitionedCallҐ#res_block_1/StatefulPartitionedCallЦ
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputinput_conv_34128317input_conv_34128319input_conv_34128321input_conv_34128323input_conv_34128325input_conv_34128327input_conv_34128329input_conv_34128331input_conv_34128333input_conv_34128335input_conv_34128337input_conv_34128339*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_341254752$
"input_conv/StatefulPartitionedCallџ
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_34128342res_block_0_34128344res_block_0_34128346res_block_0_34128348res_block_0_34128350res_block_0_34128352res_block_0_34128354res_block_0_34128356res_block_0_34128358res_block_0_34128360res_block_0_34128362res_block_0_34128364res_block_0_34128366res_block_0_34128368res_block_0_34128370res_block_0_34128372res_block_0_34128374res_block_0_34128376*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_341263122%
#res_block_0/StatefulPartitionedCall÷
tf.__operators__.add/AddV2AddV2+input_conv/StatefulPartitionedCall:output:0,res_block_0/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
tf.__operators__.add/AddV2х
relu_0/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_relu_0_layer_call_and_return_conditional_losses_341280392
relu_0/PartitionedCallѕ
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_34128381res_block_1_34128383res_block_1_34128385res_block_1_34128387res_block_1_34128389res_block_1_34128391res_block_1_34128393res_block_1_34128395res_block_1_34128397res_block_1_34128399res_block_1_34128401res_block_1_34128403res_block_1_34128405res_block_1_34128407res_block_1_34128409res_block_1_34128411res_block_1_34128413res_block_1_34128415*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_341271612%
#res_block_1/StatefulPartitionedCallќ
tf.__operators__.add_1/AddV2AddV2relu_0/PartitionedCall:output:0,res_block_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
tf.__operators__.add_1/AddV2ч
relu_1/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_relu_1_layer_call_and_return_conditional_losses_341281722
relu_1/PartitionedCallЛ
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_34128420feature_34128422feature_34128424feature_34128426feature_34128428feature_34128430*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_341274922!
feature/StatefulPartitionedCallІ
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/permе
 tf.compat.v1.transpose/transpose	Transpose(feature/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:€€€€€€€€€R2"
 tf.compat.v1.transpose/transposeУ
feature_linear/PartitionedCallPartitionedCall$tf.compat.v1.transpose/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_feature_linear_layer_call_and_return_conditional_losses_341282332 
feature_linear/PartitionedCallѓ
#output_conv/StatefulPartitionedCallStatefulPartitionedCall'feature_linear/PartitionedCall:output:0output_conv_34128436output_conv_34128438output_conv_34128440output_conv_34128442output_conv_34128444output_conv_34128446*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_output_conv_layer_call_and_return_conditional_losses_341278122%
#output_conv/StatefulPartitionedCallР
output_linear/PartitionedCallPartitionedCall,output_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_output_linear_layer_call_and_return_conditional_losses_341282922
output_linear/PartitionedCallш
softmax/PartitionedCallPartitionedCall&output_linear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_341283052
softmax/PartitionedCall≠
IdentityIdentity softmax/PartitionedCall:output:0 ^feature/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall$^res_block_0/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*°
_input_shapesП
М:€€€€€€€€€ь::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
feature/StatefulPartitionedCallfeature/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#output_conv/StatefulPartitionedCall#output_conv/StatefulPartitionedCall2J
#res_block_0/StatefulPartitionedCall#res_block_0/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:W S
0
_output_shapes
:€€€€€€€€€ь

_user_specified_nameinput
Ь/
£
I__inference_res_block_0_layer_call_and_return_conditional_losses_34126172
conv2d_2_input
conv2d_2_34126127
conv2d_2_34126129"
batch_normalization_2_34126132"
batch_normalization_2_34126134"
batch_normalization_2_34126136"
batch_normalization_2_34126138
conv2d_3_34126142
conv2d_3_34126144"
batch_normalization_3_34126147"
batch_normalization_3_34126149"
batch_normalization_3_34126151"
batch_normalization_3_34126153
conv2d_4_34126157
conv2d_4_34126159"
batch_normalization_4_34126162"
batch_normalization_4_34126164"
batch_normalization_4_34126166"
batch_normalization_4_34126168
identityИҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallѓ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_34126127conv2d_2_34126129*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_341258162"
 conv2d_2/StatefulPartitionedCallѕ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_34126132batch_normalization_2_34126134batch_normalization_2_34126136batch_normalization_2_34126138*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_341258672/
-batch_normalization_2/StatefulPartitionedCallЯ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_341259082
activation_2/PartitionedCall∆
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_34126142conv2d_3_34126144*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_341259262"
 conv2d_3/StatefulPartitionedCallѕ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_34126147batch_normalization_3_34126149batch_normalization_3_34126151batch_normalization_3_34126153*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_341259772/
-batch_normalization_3/StatefulPartitionedCallЯ
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_3_layer_call_and_return_conditional_losses_341260182
activation_3/PartitionedCall∆
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_34126157conv2d_4_34126159*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_341260362"
 conv2d_4/StatefulPartitionedCallѕ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_34126162batch_normalization_4_34126164batch_normalization_4_34126166batch_normalization_4_34126168*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_341260872/
-batch_normalization_4/StatefulPartitionedCallЛ
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€R@
(
_user_specified_nameconv2d_2_input
Ў
f
J__inference_activation_5_layer_call_and_return_conditional_losses_34131682

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ъ
ъ
.__inference_res_block_0_layer_call_fn_34130139

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
identityИҐStatefulPartitionedCallЁ
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
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_341262232
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34125691

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ў
f
J__inference_activation_5_layer_call_and_return_conditional_losses_34126867

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34131885

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
–
Ы
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34132086

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3≠
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34131589

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
м	
я
F__inference_conv2d_5_layer_call_and_return_conditional_losses_34131391

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
н	
я
F__inference_conv2d_8_layer_call_and_return_conditional_losses_34127314

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34125849

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
м	
я
F__inference_conv2d_7_layer_call_and_return_conditional_losses_34131697

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ќ
ф
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34130663

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3н
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Д/
Ы
I__inference_res_block_0_layer_call_and_return_conditional_losses_34126223

inputs
conv2d_2_34126178
conv2d_2_34126180"
batch_normalization_2_34126183"
batch_normalization_2_34126185"
batch_normalization_2_34126187"
batch_normalization_2_34126189
conv2d_3_34126193
conv2d_3_34126195"
batch_normalization_3_34126198"
batch_normalization_3_34126200"
batch_normalization_3_34126202"
batch_normalization_3_34126204
conv2d_4_34126208
conv2d_4_34126210"
batch_normalization_4_34126213"
batch_normalization_4_34126215"
batch_normalization_4_34126217"
batch_normalization_4_34126219
identityИҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallІ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_34126178conv2d_2_34126180*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_341258162"
 conv2d_2/StatefulPartitionedCallѕ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_34126183batch_normalization_2_34126185batch_normalization_2_34126187batch_normalization_2_34126189*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_341258492/
-batch_normalization_2/StatefulPartitionedCallЯ
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_341259082
activation_2/PartitionedCall∆
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0conv2d_3_34126193conv2d_3_34126195*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_341259262"
 conv2d_3/StatefulPartitionedCallѕ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_34126198batch_normalization_3_34126200batch_normalization_3_34126202batch_normalization_3_34126204*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_341259592/
-batch_normalization_3/StatefulPartitionedCallЯ
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_3_layer_call_and_return_conditional_losses_341260182
activation_3/PartitionedCall∆
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_34126208conv2d_4_34126210*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_341260362"
 conv2d_4/StatefulPartitionedCallѕ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_34126213batch_normalization_4_34126215batch_normalization_4_34126217batch_normalization_4_34126219*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_341260692/
-batch_normalization_4/StatefulPartitionedCallЛ
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
≤
В
.__inference_res_block_1_layer_call_fn_34127111
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
identityИҐStatefulPartitionedCallе
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
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_341270722
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€R@
(
_user_specified_nameconv2d_5_input
м	
я
F__inference_conv2d_2_layer_call_and_return_conditional_losses_34130942

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
“
`
D__inference_relu_1_layer_call_and_return_conditional_losses_34130409

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34125867

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
а
Ђ
8__inference_batch_normalization_7_layer_call_fn_34131755

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_341269182
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
а
Ђ
8__inference_batch_normalization_2_layer_call_fn_34131000

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_341258492
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
э!
щ
H__inference_input_conv_layer_call_and_return_conditional_losses_34125475

inputs
conv2d_34125444
conv2d_34125446 
batch_normalization_34125449 
batch_normalization_34125451 
batch_normalization_34125453 
batch_normalization_34125455
conv2d_1_34125459
conv2d_1_34125461"
batch_normalization_1_34125464"
batch_normalization_1_34125466"
batch_normalization_1_34125468"
batch_normalization_1_34125470
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallЯ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_34125444conv2d_34125446*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_341251302 
conv2d/StatefulPartitionedCallЅ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_34125449batch_normalization_34125451batch_normalization_34125453batch_normalization_34125455*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_341251812-
+batch_normalization/StatefulPartitionedCallЩ
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_341252222
activation/PartitionedCallƒ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_34125459conv2d_1_34125461*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_341252402"
 conv2d_1/StatefulPartitionedCallѕ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_34125464batch_normalization_1_34125466batch_normalization_1_34125468batch_normalization_1_34125470*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_341252912/
-batch_normalization_1/StatefulPartitionedCallЯ
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_341253322
activation_1/PartitionedCall£
IdentityIdentity%activation_1/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€ь::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ь
 
_user_specified_nameinputs
ц
h
L__inference_feature_linear_layer_call_and_return_conditional_losses_34130502

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
ƒ
ч
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34127600

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ь
ч
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34132040

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ъ
ъ
.__inference_res_block_1_layer_call_fn_34130363

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
identityИҐStatefulPartitionedCallЁ
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
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_341270722
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
K
/__inference_activation_6_layer_call_fn_34131983

inputs
identity’
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_6_layer_call_and_return_conditional_losses_341274062
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
–
Ы
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34127569

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1к
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<2
FusedBatchNormV3≠
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValueї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≤
В
.__inference_res_block_0_layer_call_fn_34126262
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
identityИҐStatefulPartitionedCallе
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
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_341262232
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€R@
(
_user_specified_nameconv2d_2_input
Ь/
£
I__inference_res_block_1_layer_call_and_return_conditional_losses_34127021
conv2d_5_input
conv2d_5_34126976
conv2d_5_34126978"
batch_normalization_5_34126981"
batch_normalization_5_34126983"
batch_normalization_5_34126985"
batch_normalization_5_34126987
conv2d_6_34126991
conv2d_6_34126993"
batch_normalization_6_34126996"
batch_normalization_6_34126998"
batch_normalization_6_34127000"
batch_normalization_6_34127002
conv2d_7_34127006
conv2d_7_34127008"
batch_normalization_7_34127011"
batch_normalization_7_34127013"
batch_normalization_7_34127015"
batch_normalization_7_34127017
identityИҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐ-batch_normalization_7/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallѓ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_34126976conv2d_5_34126978*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_341266652"
 conv2d_5/StatefulPartitionedCallѕ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_34126981batch_normalization_5_34126983batch_normalization_5_34126985batch_normalization_5_34126987*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_341267162/
-batch_normalization_5/StatefulPartitionedCallЯ
activation_4/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_4_layer_call_and_return_conditional_losses_341267572
activation_4/PartitionedCall∆
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_34126991conv2d_6_34126993*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_341267752"
 conv2d_6/StatefulPartitionedCallѕ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_34126996batch_normalization_6_34126998batch_normalization_6_34127000batch_normalization_6_34127002*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_341268262/
-batch_normalization_6/StatefulPartitionedCallЯ
activation_5/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_5_layer_call_and_return_conditional_losses_341268672
activation_5/PartitionedCall∆
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_34127006conv2d_7_34127008*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_7_layer_call_and_return_conditional_losses_341268852"
 conv2d_7/StatefulPartitionedCallѕ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_34127011batch_normalization_7_34127013batch_normalization_7_34127015batch_normalization_7_34127017*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_341269362/
-batch_normalization_7/StatefulPartitionedCallЛ
IdentityIdentity6batch_normalization_7/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€R@
(
_user_specified_nameconv2d_5_input
Д
ї
*__inference_feature_layer_call_fn_34130481

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCallґ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_341274562
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R@::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
р
К
I__inference_output_conv_layer_call_and_return_conditional_losses_34127735
conv2d_11_input
conv2d_11_34127636
conv2d_11_34127638#
batch_normalization_11_34127705#
batch_normalization_11_34127707#
batch_normalization_11_34127709#
batch_normalization_11_34127711
identityИҐ.batch_normalization_11/StatefulPartitionedCallҐ!conv2d_11/StatefulPartitionedCallµ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCallconv2d_11_inputconv2d_11_34127636conv2d_11_34127638*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_341276252#
!conv2d_11/StatefulPartitionedCall’
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_11_34127705batch_normalization_11_34127707batch_normalization_11_34127709batch_normalization_11_34127711*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *]
fXRV
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3412766020
.batch_normalization_11/StatefulPartitionedCallП
reshape_1/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_341277262
reshape_1/PartitionedCallЋ
IdentityIdentity"reshape_1/PartitionedCall:output:0/^batch_normalization_11/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R::::::2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall:` \
/
_output_shapes
:€€€€€€€€€R
)
_user_specified_nameconv2d_11_input
√
ц
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34131275

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34126509

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
«
M
1__inference_feature_linear_layer_call_fn_34130507

inputs
identity„
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_feature_linear_layer_call_and_return_conditional_losses_341282332
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
®
©
6__inference_batch_normalization_layer_call_fn_34130694

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_341249742
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Й
ф
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34130725

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1–
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:€€€€€€€€€®А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3№
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:€€€€€€€€€®А2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:€€€€€€€€€®А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:€€€€€€€€€®А
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34125660

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ў
f
J__inference_activation_3_layer_call_and_return_conditional_losses_34126018

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
•
L
0__inference_output_linear_layer_call_fn_34130616

inputs
identityќ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_output_linear_layer_call_and_return_conditional_losses_341282922
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
а
ђ
9__inference_batch_normalization_11_layer_call_fn_34132053

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *]
fXRV
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_341276602
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
д
©
6__inference_batch_normalization_layer_call_fn_34130769

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_341251812
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€®А2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:€€€€€€€€€®А::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€®А
 
_user_specified_nameinputs
а
Ђ
8__inference_batch_normalization_3_layer_call_fn_34131215

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_341259592
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ў
f
J__inference_activation_6_layer_call_and_return_conditional_losses_34131978

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
≠
H
,__inference_reshape_1_layer_call_fn_34132147

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_341277262
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34130816

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
™G
Р
E__inference_model_1_layer_call_and_return_conditional_losses_34128314	
input
input_conv_34127889
input_conv_34127891
input_conv_34127893
input_conv_34127895
input_conv_34127897
input_conv_34127899
input_conv_34127901
input_conv_34127903
input_conv_34127905
input_conv_34127907
input_conv_34127909
input_conv_34127911
res_block_0_34127996
res_block_0_34127998
res_block_0_34128000
res_block_0_34128002
res_block_0_34128004
res_block_0_34128006
res_block_0_34128008
res_block_0_34128010
res_block_0_34128012
res_block_0_34128014
res_block_0_34128016
res_block_0_34128018
res_block_0_34128020
res_block_0_34128022
res_block_0_34128024
res_block_0_34128026
res_block_0_34128028
res_block_0_34128030
res_block_1_34128129
res_block_1_34128131
res_block_1_34128133
res_block_1_34128135
res_block_1_34128137
res_block_1_34128139
res_block_1_34128141
res_block_1_34128143
res_block_1_34128145
res_block_1_34128147
res_block_1_34128149
res_block_1_34128151
res_block_1_34128153
res_block_1_34128155
res_block_1_34128157
res_block_1_34128159
res_block_1_34128161
res_block_1_34128163
feature_34128214
feature_34128216
feature_34128218
feature_34128220
feature_34128222
feature_34128224
output_conv_34128275
output_conv_34128277
output_conv_34128279
output_conv_34128281
output_conv_34128283
output_conv_34128285
identityИҐfeature/StatefulPartitionedCallҐ"input_conv/StatefulPartitionedCallҐ#output_conv/StatefulPartitionedCallҐ#res_block_0/StatefulPartitionedCallҐ#res_block_1/StatefulPartitionedCallЦ
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputinput_conv_34127889input_conv_34127891input_conv_34127893input_conv_34127895input_conv_34127897input_conv_34127899input_conv_34127901input_conv_34127903input_conv_34127905input_conv_34127907input_conv_34127909input_conv_34127911*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_input_conv_layer_call_and_return_conditional_losses_341254122$
"input_conv/StatefulPartitionedCallџ
#res_block_0/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:0res_block_0_34127996res_block_0_34127998res_block_0_34128000res_block_0_34128002res_block_0_34128004res_block_0_34128006res_block_0_34128008res_block_0_34128010res_block_0_34128012res_block_0_34128014res_block_0_34128016res_block_0_34128018res_block_0_34128020res_block_0_34128022res_block_0_34128024res_block_0_34128026res_block_0_34128028res_block_0_34128030*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_0_layer_call_and_return_conditional_losses_341262232%
#res_block_0/StatefulPartitionedCall÷
tf.__operators__.add/AddV2AddV2+input_conv/StatefulPartitionedCall:output:0,res_block_0/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
tf.__operators__.add/AddV2х
relu_0/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_relu_0_layer_call_and_return_conditional_losses_341280392
relu_0/PartitionedCallѕ
#res_block_1/StatefulPartitionedCallStatefulPartitionedCallrelu_0/PartitionedCall:output:0res_block_1_34128129res_block_1_34128131res_block_1_34128133res_block_1_34128135res_block_1_34128137res_block_1_34128139res_block_1_34128141res_block_1_34128143res_block_1_34128145res_block_1_34128147res_block_1_34128149res_block_1_34128151res_block_1_34128153res_block_1_34128155res_block_1_34128157res_block_1_34128159res_block_1_34128161res_block_1_34128163*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_res_block_1_layer_call_and_return_conditional_losses_341270722%
#res_block_1/StatefulPartitionedCallќ
tf.__operators__.add_1/AddV2AddV2relu_0/PartitionedCall:output:0,res_block_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
tf.__operators__.add_1/AddV2ч
relu_1/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_relu_1_layer_call_and_return_conditional_losses_341281722
relu_1/PartitionedCallЛ
feature/StatefulPartitionedCallStatefulPartitionedCallrelu_1/PartitionedCall:output:0feature_34128214feature_34128216feature_34128218feature_34128220feature_34128222feature_34128224*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_341274562!
feature/StatefulPartitionedCallІ
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%tf.compat.v1.transpose/transpose/permе
 tf.compat.v1.transpose/transpose	Transpose(feature/StatefulPartitionedCall:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:€€€€€€€€€R2"
 tf.compat.v1.transpose/transposeУ
feature_linear/PartitionedCallPartitionedCall$tf.compat.v1.transpose/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_feature_linear_layer_call_and_return_conditional_losses_341282332 
feature_linear/PartitionedCall≠
#output_conv/StatefulPartitionedCallStatefulPartitionedCall'feature_linear/PartitionedCall:output:0output_conv_34128275output_conv_34128277output_conv_34128279output_conv_34128281output_conv_34128283output_conv_34128285*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_output_conv_layer_call_and_return_conditional_losses_341277762%
#output_conv/StatefulPartitionedCallР
output_linear/PartitionedCallPartitionedCall,output_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *T
fORM
K__inference_output_linear_layer_call_and_return_conditional_losses_341282922
output_linear/PartitionedCallш
softmax/PartitionedCallPartitionedCall&output_linear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_341283052
softmax/PartitionedCall≠
IdentityIdentity softmax/PartitionedCall:output:0 ^feature/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall$^output_conv/StatefulPartitionedCall$^res_block_0/StatefulPartitionedCall$^res_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*°
_input_shapesП
М:€€€€€€€€€ь::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2B
feature/StatefulPartitionedCallfeature/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2J
#output_conv/StatefulPartitionedCall#output_conv/StatefulPartitionedCall2J
#res_block_0/StatefulPartitionedCall#res_block_0/StatefulPartitionedCall2J
#res_block_1/StatefulPartitionedCall#res_block_1/StatefulPartitionedCall:W S
0
_output_shapes
:€€€€€€€€€ь

_user_specified_nameinput
в
ђ
9__inference_batch_normalization_11_layer_call_fn_34132066

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *]
fXRV
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_341276782
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34131651

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ьV
ё
I__inference_res_block_0_layer_call_and_return_conditional_losses_34130098

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
identityИҐ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґ5batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_3/ReadVariableOpҐ&batch_normalization_3/ReadVariableOp_1Ґ5batch_normalization_4/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_4/ReadVariableOpҐ&batch_normalization_4/ReadVariableOp_1Ґconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOpҐconv2d_3/BiasAdd/ReadVariableOpҐconv2d_3/Conv2D/ReadVariableOpҐconv2d_4/BiasAdd/ReadVariableOpҐconv2d_4/Conv2D/ReadVariableOp∞
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpЊ
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
conv2d_2/Conv2DІ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpђ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
conv2d_2/BiasAddґ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOpЉ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1й
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3Ф
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
activation_2/Relu∞
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp„
conv2d_3/Conv2DConv2Dactivation_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
conv2d_3/Conv2DІ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpђ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
conv2d_3/BiasAddґ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOpЉ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1й
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3Ф
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
activation_3/Relu∞
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp„
conv2d_4/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
conv2d_4/Conv2DІ
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOpђ
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
conv2d_4/BiasAddґ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOpЉ
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1й
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3Х
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::2n
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
:€€€€€€€€€R@
 
_user_specified_nameinputs
И
Б
,__inference_conv2d_11_layer_call_fn_34132002

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_341276252
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
ьV
ё
I__inference_res_block_1_layer_call_and_return_conditional_losses_34130256

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
identityИҐ5batch_normalization_5/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_5/ReadVariableOpҐ&batch_normalization_5/ReadVariableOp_1Ґ5batch_normalization_6/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_6/ReadVariableOpҐ&batch_normalization_6/ReadVariableOp_1Ґ5batch_normalization_7/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_7/ReadVariableOpҐ&batch_normalization_7/ReadVariableOp_1Ґconv2d_5/BiasAdd/ReadVariableOpҐconv2d_5/Conv2D/ReadVariableOpҐconv2d_6/BiasAdd/ReadVariableOpҐconv2d_6/Conv2D/ReadVariableOpҐconv2d_7/BiasAdd/ReadVariableOpҐconv2d_7/Conv2D/ReadVariableOp∞
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_5/Conv2D/ReadVariableOpЊ
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
conv2d_5/Conv2DІ
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOpђ
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
conv2d_5/BiasAddґ
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOpЉ
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1й
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3Ф
activation_4/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
activation_4/Relu∞
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_6/Conv2D/ReadVariableOp„
conv2d_6/Conv2DConv2Dactivation_4/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
conv2d_6/Conv2DІ
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOpђ
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
conv2d_6/BiasAddґ
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOpЉ
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1й
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3Ф
activation_5/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
activation_5/Relu∞
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp„
conv2d_7/Conv2DConv2Dactivation_5/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingSAME*
strides
2
conv2d_7/Conv2DІ
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOpђ
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
conv2d_7/BiasAddґ
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_7/ReadVariableOpЉ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_7/ReadVariableOp_1й
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3Х
IdentityIdentity*batch_normalization_7/FusedBatchNormV3:y:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::2n
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
:€€€€€€€€€R@
 
_user_specified_nameinputs
”
х
E__inference_feature_layer_call_and_return_conditional_losses_34127492

inputs
conv2d_8_34127476
conv2d_8_34127478"
batch_normalization_8_34127481"
batch_normalization_8_34127483"
batch_normalization_8_34127485"
batch_normalization_8_34127487
identityИҐ-batch_normalization_8/StatefulPartitionedCallҐ conv2d_8/StatefulPartitionedCallІ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_34127476conv2d_8_34127478*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_8_layer_call_and_return_conditional_losses_341273142"
 conv2d_8/StatefulPartitionedCallѕ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_8_34127481batch_normalization_8_34127483batch_normalization_8_34127485batch_normalization_8_34127487*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_341273652/
-batch_normalization_8/StatefulPartitionedCallЯ
activation_6/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_6_layer_call_and_return_conditional_losses_341274062
activation_6/PartitionedCall‘
IdentityIdentity%activation_6/PartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R@::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Х
»
.__inference_output_conv_layer_call_fn_34127791
conv2d_11_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallconv2d_11_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_output_conv_layer_call_and_return_conditional_losses_341277762
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:€€€€€€€€€R
)
_user_specified_nameconv2d_11_input
к:
у	
H__inference_input_conv_layer_call_and_return_conditional_losses_34129908

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
identityИҐ3batch_normalization/FusedBatchNormV3/ReadVariableOpҐ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ"batch_normalization/ReadVariableOpҐ$batch_normalization/ReadVariableOp_1Ґ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_1/ReadVariableOpҐ&batch_normalization_1/ReadVariableOp_1Ґconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpЂ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype02
conv2d/Conv2D/ReadVariableOpї
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€®А*
paddingVALID*
strides
2
conv2d/Conv2DҐ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
conv2d/BiasAdd/ReadVariableOp¶
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€®А2
conv2d/BiasAdd±
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"batch_normalization/ReadVariableOpЈ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:А*
dtype02&
$batch_normalization/ReadVariableOp_1д
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpк
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ў
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:€€€€€€€€€®А:А:А:А:А:*
epsilon%oГ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3Р
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:€€€€€€€€€®А2
activation/Relu±
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp÷
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@*
paddingVALID*
strides
2
conv2d_1/Conv2DІ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
conv2d_1/BiasAddґ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_1/ReadVariableOpЉ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_1/ReadVariableOp_1й
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3Ф
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€R@2
activation_1/Reluщ
IdentityIdentityactivation_1/Relu:activations:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:€€€€€€€€€ь::::::::::::2j
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
:€€€€€€€€€ь
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34130834

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34125074

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34130969

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ь
√
*__inference_feature_layer_call_fn_34127471
conv2d_8_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_feature_layer_call_and_return_conditional_losses_341274562
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€R@::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€R@
(
_user_specified_nameconv2d_8_input
ы
ц
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34125977

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34126609

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ь/
£
I__inference_res_block_1_layer_call_and_return_conditional_losses_34126973
conv2d_5_input
conv2d_5_34126676
conv2d_5_34126678"
batch_normalization_5_34126743"
batch_normalization_5_34126745"
batch_normalization_5_34126747"
batch_normalization_5_34126749
conv2d_6_34126786
conv2d_6_34126788"
batch_normalization_6_34126853"
batch_normalization_6_34126855"
batch_normalization_6_34126857"
batch_normalization_6_34126859
conv2d_7_34126896
conv2d_7_34126898"
batch_normalization_7_34126963"
batch_normalization_7_34126965"
batch_normalization_7_34126967"
batch_normalization_7_34126969
identityИҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐ-batch_normalization_7/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallѓ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_34126676conv2d_5_34126678*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_341266652"
 conv2d_5/StatefulPartitionedCallѕ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_34126743batch_normalization_5_34126745batch_normalization_5_34126747batch_normalization_5_34126749*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_341266982/
-batch_normalization_5/StatefulPartitionedCallЯ
activation_4/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_4_layer_call_and_return_conditional_losses_341267572
activation_4/PartitionedCall∆
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0conv2d_6_34126786conv2d_6_34126788*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_341267752"
 conv2d_6/StatefulPartitionedCallѕ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_34126853batch_normalization_6_34126855batch_normalization_6_34126857batch_normalization_6_34126859*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_341268082/
-batch_normalization_6/StatefulPartitionedCallЯ
activation_5/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_5_layer_call_and_return_conditional_losses_341268672
activation_5/PartitionedCall∆
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0conv2d_7_34126896conv2d_7_34126898*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_7_layer_call_and_return_conditional_losses_341268852"
 conv2d_7/StatefulPartitionedCallѕ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_7_34126963batch_normalization_7_34126965batch_normalization_7_34126967batch_normalization_7_34126969*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_341269182/
-batch_normalization_7/StatefulPartitionedCallЛ
IdentityIdentity6batch_normalization_7/StatefulPartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*v
_input_shapese
c:€€€€€€€€€R@::::::::::::::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:_ [
/
_output_shapes
:€€€€€€€€€R@
(
_user_specified_nameconv2d_5_input
а
Ђ
8__inference_batch_normalization_1_layer_call_fn_34130909

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_341252732
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
®
Ђ
8__inference_batch_normalization_7_layer_call_fn_34131830

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_341266402
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34131202

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ў
f
J__inference_activation_3_layer_call_and_return_conditional_losses_34131233

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ы
ц
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34131724

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€R@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Џ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
Ў
f
J__inference_activation_1_layer_call_and_return_conditional_losses_34125332

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
ї
a
E__inference_softmax_layer_call_and_return_conditional_losses_34128305

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:€€€€€€€€€2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
“
`
D__inference_relu_0_layer_call_and_return_conditional_losses_34130185

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
®
Ђ
8__inference_batch_normalization_6_layer_call_fn_34131615

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_341265402
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34131804

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
а
Ђ
8__inference_batch_normalization_8_layer_call_fn_34131911

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_341273652
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€R::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34125591

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ж
А
+__inference_conv2d_6_layer_call_fn_34131553

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_341267752
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€R@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34125560

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ў
f
J__inference_activation_4_layer_call_and_return_conditional_losses_34131529

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€R@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs
√
ц
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34131947

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3м
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
«
I
-__inference_activation_layer_call_fn_34130779

inputs
identity’
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€®А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_341252222
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€®А2

Identity"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€®А:Y U
1
_output_shapes
:€€€€€€€€€®А
 
_user_specified_nameinputs
®
Ђ
8__inference_batch_normalization_3_layer_call_fn_34131153

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_341256602
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
√
K
/__inference_activation_1_layer_call_fn_34130932

inputs
identity’
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€R@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_341253322
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€R@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€R@:W S
/
_output_shapes
:€€€€€€€€€R@
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ѓ
serving_defaultЫ
@
input7
serving_default_input:0€€€€€€€€€ь;
softmax0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:Ъж
“Ы
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
layer_with_weights-4
layer-11
layer-12
layer-13
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
м_default_save_signature
н__call__
+о&call_and_return_all_conditional_losses"«Ч
_tf_keras_network™Ч{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "input_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "name": "input_conv", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "res_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "res_block_0", "inbound_nodes": [[["input_conv", 1, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": false, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["input_conv", 1, 0, {"y": ["res_block_0", 1, 0], "name": null}]]}, {"class_name": "Activation", "config": {"name": "relu_0", "trainable": false, "dtype": "float32", "activation": "relu"}, "name": "relu_0", "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "res_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "res_block_1", "inbound_nodes": [[["relu_0", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": false, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["relu_0", 0, 0, {"y": ["res_block_1", 1, 0], "name": null}]]}, {"class_name": "Activation", "config": {"name": "relu_1", "trainable": false, "dtype": "float32", "activation": "relu"}, "name": "relu_1", "inbound_nodes": [[["tf.__operators__.add_1", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "feature", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "name": "feature", "inbound_nodes": [[["relu_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.transpose", "trainable": false, "dtype": "float32", "function": "compat.v1.transpose"}, "name": "tf.compat.v1.transpose", "inbound_nodes": [["feature", 1, 0, {"perm": {"class_name": "__tuple__", "items": [0, 1, 3, 2]}}]]}, {"class_name": "Activation", "config": {"name": "feature_linear", "trainable": false, "dtype": "float32", "activation": "linear"}, "name": "feature_linear", "inbound_nodes": [[["tf.compat.v1.transpose", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "output_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_11_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}}]}, "name": "output_conv", "inbound_nodes": [[["feature_linear", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "output_linear", "trainable": true, "dtype": "float32", "activation": "linear"}, "name": "output_linear", "inbound_nodes": [[["output_conv", 1, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "activation": "softmax"}, "name": "softmax", "inbound_nodes": [[["output_linear", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["softmax", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 508, 8]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "input_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "name": "input_conv", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "res_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "res_block_0", "inbound_nodes": [[["input_conv", 1, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": false, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["input_conv", 1, 0, {"y": ["res_block_0", 1, 0], "name": null}]]}, {"class_name": "Activation", "config": {"name": "relu_0", "trainable": false, "dtype": "float32", "activation": "relu"}, "name": "relu_0", "inbound_nodes": [[["tf.__operators__.add", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "res_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "res_block_1", "inbound_nodes": [[["relu_0", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": false, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["relu_0", 0, 0, {"y": ["res_block_1", 1, 0], "name": null}]]}, {"class_name": "Activation", "config": {"name": "relu_1", "trainable": false, "dtype": "float32", "activation": "relu"}, "name": "relu_1", "inbound_nodes": [[["tf.__operators__.add_1", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "feature", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "name": "feature", "inbound_nodes": [[["relu_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.transpose", "trainable": false, "dtype": "float32", "function": "compat.v1.transpose"}, "name": "tf.compat.v1.transpose", "inbound_nodes": [["feature", 1, 0, {"perm": {"class_name": "__tuple__", "items": [0, 1, 3, 2]}}]]}, {"class_name": "Activation", "config": {"name": "feature_linear", "trainable": false, "dtype": "float32", "activation": "linear"}, "name": "feature_linear", "inbound_nodes": [[["tf.compat.v1.transpose", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "output_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_11_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}}]}, "name": "output_conv", "inbound_nodes": [[["feature_linear", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "output_linear", "trainable": true, "dtype": "float32", "activation": "linear"}, "name": "output_linear", "inbound_nodes": [[["output_conv", 1, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "activation": "softmax"}, "name": "softmax", "inbound_nodes": [[["output_linear", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["softmax", 0, 0]]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": {"class_name": "CosineDecay", "config": {"initial_learning_rate": 0.001, "decay_steps": 26200, "alpha": 0.0, "name": null}}, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ч"ф
_tf_keras_input_layer‘{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
≥5
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
trainable_variables
regularization_losses
	variables
	keras_api
п__call__
+р&call_and_return_all_conditional_losses"м2
_tf_keras_sequentialЌ2{"class_name": "Sequential", "name": "input_conv", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "input_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 508, 8]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "input_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 508, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}]}}}
ЈI
layer_with_weights-0
layer-0
 layer_with_weights-1
 layer-1
!layer-2
"layer_with_weights-2
"layer-3
#layer_with_weights-3
#layer-4
$layer-5
%layer_with_weights-4
%layer-6
&layer_with_weights-5
&layer-7
'trainable_variables
(regularization_losses
)	variables
*	keras_api
с__call__
+т&call_and_return_all_conditional_losses"ҐF
_tf_keras_sequentialГF{"class_name": "Sequential", "name": "res_block_0", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "res_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "res_block_0", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}}}
ф
+	keras_api"в
_tf_keras_layer»{"class_name": "TFOpLambda", "name": "tf.__operators__.add", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add", "trainable": false, "dtype": "float32", "function": "__operators__.add"}}
Ќ
,trainable_variables
-regularization_losses
.	variables
/	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"Љ
_tf_keras_layerҐ{"class_name": "Activation", "name": "relu_0", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu_0", "trainable": false, "dtype": "float32", "activation": "relu"}}
ЈI
0layer_with_weights-0
0layer-0
1layer_with_weights-1
1layer-1
2layer-2
3layer_with_weights-2
3layer-3
4layer_with_weights-3
4layer-4
5layer-5
6layer_with_weights-4
6layer-6
7layer_with_weights-5
7layer-7
8trainable_variables
9regularization_losses
:	variables
;	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"ҐF
_tf_keras_sequentialГF{"class_name": "Sequential", "name": "res_block_1", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "res_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "res_block_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}}}
ш
<	keras_api"ж
_tf_keras_layerћ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_1", "trainable": false, "dtype": "float32", "function": "__operators__.add"}}
Ќ
=trainable_variables
>regularization_losses
?	variables
@	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"Љ
_tf_keras_layerҐ{"class_name": "Activation", "name": "relu_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu_1", "trainable": false, "dtype": "float32", "activation": "relu"}}
≤
Alayer_with_weights-0
Alayer-0
Blayer_with_weights-1
Blayer-1
Clayer-2
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses"∆
_tf_keras_sequentialІ{"class_name": "Sequential", "name": "feature", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "feature", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "feature", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 82, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_8_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}]}}}
ъ
H	keras_api"и
_tf_keras_layerќ{"class_name": "TFOpLambda", "name": "tf.compat.v1.transpose", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.transpose", "trainable": false, "dtype": "float32", "function": "compat.v1.transpose"}}
я
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "Activation", "name": "feature_linear", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "feature_linear", "trainable": false, "dtype": "float32", "activation": "linear"}}
ь
Mlayer_with_weights-0
Mlayer-0
Nlayer_with_weights-1
Nlayer-1
Olayer-2
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"Р
_tf_keras_sequentialс{"class_name": "Sequential", "name": "output_conv", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "output_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_11_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 82}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 8, 82]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "output_conv", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 8, 82]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_11_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}}]}}}
џ
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
€__call__
+А&call_and_return_all_conditional_losses" 
_tf_keras_layer∞{"class_name": "Activation", "name": "output_linear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output_linear", "trainable": true, "dtype": "float32", "activation": "linear"}}
–
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"њ
_tf_keras_layer•{"class_name": "Activation", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "activation": "softmax"}}
Р
\iter

]beta_1

^beta_2
	_decay`mдamеbmжcmз`vиavйbvкcvл"
	optimizer
<
`0
a1
b2
c3"
trackable_list_wrapper
 "
trackable_list_wrapper
Т
d0
e1
f2
g3
h4
i5
j6
k7
l8
m9
n10
o11
p12
q13
r14
s15
t16
u17
v18
w19
x20
y21
z22
{23
|24
}25
~26
27
А28
Б29
В30
Г31
Д32
Е33
Ж34
З35
И36
Й37
К38
Л39
М40
Н41
О42
П43
Р44
С45
Т46
У47
Ф48
Х49
Ц50
Ч51
Ш52
Щ53
`54
a55
b56
c57
Ъ58
Ы59"
trackable_list_wrapper
”
trainable_variables
Ьnon_trainable_variables
Эmetrics
Юlayers
regularization_losses
 Яlayer_regularization_losses
†layer_metrics
	variables
н__call__
м_default_save_signature
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
-
Гserving_default"
signature_map
щ	

dkernel
ebias
°trainable_variables
Ґregularization_losses
£	variables
§	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "Conv2D", "name": "conv2d", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 508, 8]}}
¬	
	•axis
	fgamma
gbeta
hmoving_mean
imoving_variance
¶trainable_variables
Іregularization_losses
®	variables
©	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"з
_tf_keras_layerЌ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 168, 128]}}
ў
™trainable_variables
Ђregularization_losses
ђ	variables
≠	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"ƒ
_tf_keras_layer™{"class_name": "Activation", "name": "activation", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": false, "dtype": "float32", "activation": "relu"}}
А


jkernel
kbias
Ѓtrainable_variables
ѓregularization_losses
∞	variables
±	keras_api
К__call__
+Л&call_and_return_all_conditional_losses"’
_tf_keras_layerї{"class_name": "Conv2D", "name": "conv2d_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 168, 128]}}
√	
	≤axis
	lgamma
mbeta
nmoving_mean
omoving_variance
≥trainable_variables
іregularization_losses
µ	variables
ґ	keras_api
М__call__
+Н&call_and_return_all_conditional_losses"и
_tf_keras_layerќ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
Ё
Јtrainable_variables
Єregularization_losses
є	variables
Ї	keras_api
О__call__
+П&call_and_return_all_conditional_losses"»
_tf_keras_layerЃ{"class_name": "Activation", "name": "activation_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": false, "dtype": "float32", "activation": "relu"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
v
d0
e1
f2
g3
h4
i5
j6
k7
l8
m9
n10
o11"
trackable_list_wrapper
µ
trainable_variables
їnon_trainable_variables
Љmetrics
љlayers
regularization_losses
 Њlayer_regularization_losses
њlayer_metrics
	variables
п__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
ь	

pkernel
qbias
јtrainable_variables
Ѕregularization_losses
¬	variables
√	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "conv2d_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
√	
	ƒaxis
	rgamma
sbeta
tmoving_mean
umoving_variance
≈trainable_variables
∆regularization_losses
«	variables
»	keras_api
Т__call__
+У&call_and_return_all_conditional_losses"и
_tf_keras_layerќ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
Ё
…trainable_variables
 regularization_losses
Ћ	variables
ћ	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"»
_tf_keras_layerЃ{"class_name": "Activation", "name": "activation_2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": false, "dtype": "float32", "activation": "relu"}}
ь	

vkernel
wbias
Ќtrainable_variables
ќregularization_losses
ѕ	variables
–	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "conv2d_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
√	
	—axis
	xgamma
ybeta
zmoving_mean
{moving_variance
“trainable_variables
”regularization_losses
‘	variables
’	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses"и
_tf_keras_layerќ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
Ё
÷trainable_variables
„regularization_losses
Ў	variables
ў	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses"»
_tf_keras_layerЃ{"class_name": "Activation", "name": "activation_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": false, "dtype": "float32", "activation": "relu"}}
ь	

|kernel
}bias
Џtrainable_variables
џregularization_losses
№	variables
Ё	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "conv2d_4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
≈	
	ёaxis
	~gamma
beta
Аmoving_mean
Бmoving_variance
яtrainable_variables
аregularization_losses
б	variables
в	keras_api
Ю__call__
+Я&call_and_return_all_conditional_losses"и
_tf_keras_layerќ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
®
p0
q1
r2
s3
t4
u5
v6
w7
x8
y9
z10
{11
|12
}13
~14
15
А16
Б17"
trackable_list_wrapper
µ
'trainable_variables
гnon_trainable_variables
дmetrics
еlayers
(regularization_losses
 жlayer_regularization_losses
зlayer_metrics
)	variables
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
,trainable_variables
иmetrics
йnon_trainable_variables
кlayers
-regularization_losses
 лlayer_regularization_losses
мlayer_metrics
.	variables
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
ю	
Вkernel
	Гbias
нtrainable_variables
оregularization_losses
п	variables
р	keras_api
†__call__
+°&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "conv2d_5", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
«	
	сaxis

Дgamma
	Еbeta
Жmoving_mean
Зmoving_variance
тtrainable_variables
уregularization_losses
ф	variables
х	keras_api
Ґ__call__
+£&call_and_return_all_conditional_losses"и
_tf_keras_layerќ{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
Ё
цtrainable_variables
чregularization_losses
ш	variables
щ	keras_api
§__call__
+•&call_and_return_all_conditional_losses"»
_tf_keras_layerЃ{"class_name": "Activation", "name": "activation_4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": false, "dtype": "float32", "activation": "relu"}}
ю	
Иkernel
	Йbias
ъtrainable_variables
ыregularization_losses
ь	variables
э	keras_api
¶__call__
+І&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "conv2d_6", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
«	
	юaxis

Кgamma
	Лbeta
Мmoving_mean
Нmoving_variance
€trainable_variables
Аregularization_losses
Б	variables
В	keras_api
®__call__
+©&call_and_return_all_conditional_losses"и
_tf_keras_layerќ{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
Ё
Гtrainable_variables
Дregularization_losses
Е	variables
Ж	keras_api
™__call__
+Ђ&call_and_return_all_conditional_losses"»
_tf_keras_layerЃ{"class_name": "Activation", "name": "activation_5", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_5", "trainable": false, "dtype": "float32", "activation": "relu"}}
ю	
Оkernel
	Пbias
Зtrainable_variables
Иregularization_losses
Й	variables
К	keras_api
ђ__call__
+≠&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "conv2d_7", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": false, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
«	
	Лaxis

Рgamma
	Сbeta
Тmoving_mean
Уmoving_variance
Мtrainable_variables
Нregularization_losses
О	variables
П	keras_api
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses"и
_tf_keras_layerќ{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
В0
Г1
Д2
Е3
Ж4
З5
И6
Й7
К8
Л9
М10
Н11
О12
П13
Р14
С15
Т16
У17"
trackable_list_wrapper
µ
8trainable_variables
Рnon_trainable_variables
Сmetrics
Тlayers
9regularization_losses
 Уlayer_regularization_losses
Фlayer_metrics
:	variables
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
=trainable_variables
Хmetrics
Цnon_trainable_variables
Чlayers
>regularization_losses
 Шlayer_regularization_losses
Щlayer_metrics
?	variables
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
ю	
Фkernel
	Хbias
Ъtrainable_variables
Ыregularization_losses
Ь	variables
Э	keras_api
∞__call__
+±&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "conv2d_8", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_8", "trainable": false, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 64]}}
≈	
	Юaxis

Цgamma
	Чbeta
Шmoving_mean
Щmoving_variance
Яtrainable_variables
†regularization_losses
°	variables
Ґ	keras_api
≤__call__
+≥&call_and_return_all_conditional_losses"ж
_tf_keras_layerћ{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 82, 8]}}
Ё
£trainable_variables
§regularization_losses
•	variables
¶	keras_api
і__call__
+µ&call_and_return_all_conditional_losses"»
_tf_keras_layerЃ{"class_name": "Activation", "name": "activation_6", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_6", "trainable": false, "dtype": "float32", "activation": "relu"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
P
Ф0
Х1
Ц2
Ч3
Ш4
Щ5"
trackable_list_wrapper
µ
Dtrainable_variables
Іnon_trainable_variables
®metrics
©layers
Eregularization_losses
 ™layer_regularization_losses
Ђlayer_metrics
F	variables
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Itrainable_variables
ђmetrics
≠non_trainable_variables
Ѓlayers
Jregularization_losses
 ѓlayer_regularization_losses
∞layer_metrics
K	variables
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
ь	

`kernel
abias
±trainable_variables
≤regularization_losses
≥	variables
і	keras_api
ґ__call__
+Ј&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [30, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 82}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 8, 82]}}
Ѕ	
	µaxis
	bgamma
cbeta
Ъmoving_mean
Ыmoving_variance
ґtrainable_variables
Јregularization_losses
Є	variables
є	keras_api
Є__call__
+є&call_and_return_all_conditional_losses"д
_tf_keras_layer {"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 8, 1]}}
ш
Їtrainable_variables
їregularization_losses
Љ	variables
љ	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses"г
_tf_keras_layer…{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1]}}}
<
`0
a1
b2
c3"
trackable_list_wrapper
 "
trackable_list_wrapper
L
`0
a1
b2
c3
Ъ4
Ы5"
trackable_list_wrapper
µ
Ptrainable_variables
Њnon_trainable_variables
њmetrics
јlayers
Qregularization_losses
 Ѕlayer_regularization_losses
¬layer_metrics
R	variables
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ttrainable_variables
√metrics
ƒnon_trainable_variables
≈layers
Uregularization_losses
 ∆layer_regularization_losses
«layer_metrics
V	variables
€__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Xtrainable_variables
»metrics
…non_trainable_variables
 layers
Yregularization_losses
 Ћlayer_regularization_losses
ћlayer_metrics
Z	variables
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
*:(R2conv2d_11/kernel
:2conv2d_11/bias
*:(2batch_normalization_11/gamma
):'2batch_normalization_11/beta
(:&А2conv2d/kernel
:А2conv2d/bias
(:&А2batch_normalization/gamma
':%А2batch_normalization/beta
0:.А (2batch_normalization/moving_mean
4:2А (2#batch_normalization/moving_variance
*:(А@2conv2d_1/kernel
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
2:0 (2"batch_normalization_11/moving_mean
6:4 (2&batch_normalization_11/moving_variance
т
d0
e1
f2
g3
h4
i5
j6
k7
l8
m9
n10
o11
p12
q13
r14
s15
t16
u17
v18
w19
x20
y21
z22
{23
|24
}25
~26
27
А28
Б29
В30
Г31
Д32
Е33
Ж34
З35
И36
Й37
К38
Л39
М40
Н41
О42
П43
Р44
С45
Т46
У47
Ф48
Х49
Ц50
Ч51
Ш52
Щ53
Ъ54
Ы55"
trackable_list_wrapper
0
Ќ0
ќ1"
trackable_list_wrapper
Ж
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
11
12
13"
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
d0
e1"
trackable_list_wrapper
Є
°trainable_variables
ѕmetrics
–non_trainable_variables
—layers
Ґregularization_losses
 “layer_regularization_losses
”layer_metrics
£	variables
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
f0
g1
h2
i3"
trackable_list_wrapper
Є
¶trainable_variables
‘metrics
’non_trainable_variables
÷layers
Іregularization_losses
 „layer_regularization_losses
Ўlayer_metrics
®	variables
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
™trainable_variables
ўmetrics
Џnon_trainable_variables
џlayers
Ђregularization_losses
 №layer_regularization_losses
Ёlayer_metrics
ђ	variables
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
Є
Ѓtrainable_variables
ёmetrics
яnon_trainable_variables
аlayers
ѓregularization_losses
 бlayer_regularization_losses
вlayer_metrics
∞	variables
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
l0
m1
n2
o3"
trackable_list_wrapper
Є
≥trainable_variables
гmetrics
дnon_trainable_variables
еlayers
іregularization_losses
 жlayer_regularization_losses
зlayer_metrics
µ	variables
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Јtrainable_variables
иmetrics
йnon_trainable_variables
кlayers
Єregularization_losses
 лlayer_regularization_losses
мlayer_metrics
є	variables
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
v
d0
e1
f2
g3
h4
i5
j6
k7
l8
m9
n10
o11"
trackable_list_wrapper
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
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
Є
јtrainable_variables
нmetrics
оnon_trainable_variables
пlayers
Ѕregularization_losses
 рlayer_regularization_losses
сlayer_metrics
¬	variables
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
r0
s1
t2
u3"
trackable_list_wrapper
Є
≈trainable_variables
тmetrics
уnon_trainable_variables
фlayers
∆regularization_losses
 хlayer_regularization_losses
цlayer_metrics
«	variables
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
…trainable_variables
чmetrics
шnon_trainable_variables
щlayers
 regularization_losses
 ъlayer_regularization_losses
ыlayer_metrics
Ћ	variables
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
Є
Ќtrainable_variables
ьmetrics
эnon_trainable_variables
юlayers
ќregularization_losses
 €layer_regularization_losses
Аlayer_metrics
ѕ	variables
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
x0
y1
z2
{3"
trackable_list_wrapper
Є
“trainable_variables
Бmetrics
Вnon_trainable_variables
Гlayers
”regularization_losses
 Дlayer_regularization_losses
Еlayer_metrics
‘	variables
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
÷trainable_variables
Жmetrics
Зnon_trainable_variables
Иlayers
„regularization_losses
 Йlayer_regularization_losses
Кlayer_metrics
Ў	variables
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
Є
Џtrainable_variables
Лmetrics
Мnon_trainable_variables
Нlayers
џregularization_losses
 Оlayer_regularization_losses
Пlayer_metrics
№	variables
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
>
~0
1
А2
Б3"
trackable_list_wrapper
Є
яtrainable_variables
Рmetrics
Сnon_trainable_variables
Тlayers
аregularization_losses
 Уlayer_regularization_losses
Фlayer_metrics
б	variables
Ю__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
®
p0
q1
r2
s3
t4
u5
v6
w7
x8
y9
z10
{11
|12
}13
~14
15
А16
Б17"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
 1
!2
"3
#4
$5
%6
&7"
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
В0
Г1"
trackable_list_wrapper
Є
нtrainable_variables
Хmetrics
Цnon_trainable_variables
Чlayers
оregularization_losses
 Шlayer_regularization_losses
Щlayer_metrics
п	variables
†__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
Д0
Е1
Ж2
З3"
trackable_list_wrapper
Є
тtrainable_variables
Ъmetrics
Ыnon_trainable_variables
Ьlayers
уregularization_losses
 Эlayer_regularization_losses
Юlayer_metrics
ф	variables
Ґ__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
цtrainable_variables
Яmetrics
†non_trainable_variables
°layers
чregularization_losses
 Ґlayer_regularization_losses
£layer_metrics
ш	variables
§__call__
+•&call_and_return_all_conditional_losses
'•"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
И0
Й1"
trackable_list_wrapper
Є
ъtrainable_variables
§metrics
•non_trainable_variables
¶layers
ыregularization_losses
 Іlayer_regularization_losses
®layer_metrics
ь	variables
¶__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
К0
Л1
М2
Н3"
trackable_list_wrapper
Є
€trainable_variables
©metrics
™non_trainable_variables
Ђlayers
Аregularization_losses
 ђlayer_regularization_losses
≠layer_metrics
Б	variables
®__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Гtrainable_variables
Ѓmetrics
ѓnon_trainable_variables
∞layers
Дregularization_losses
 ±layer_regularization_losses
≤layer_metrics
Е	variables
™__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
О0
П1"
trackable_list_wrapper
Є
Зtrainable_variables
≥metrics
іnon_trainable_variables
µlayers
Иregularization_losses
 ґlayer_regularization_losses
Јlayer_metrics
Й	variables
ђ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
Р0
С1
Т2
У3"
trackable_list_wrapper
Є
Мtrainable_variables
Єmetrics
єnon_trainable_variables
Їlayers
Нregularization_losses
 їlayer_regularization_losses
Љlayer_metrics
О	variables
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
Є
В0
Г1
Д2
Е3
Ж4
З5
И6
Й7
К8
Л9
М10
Н11
О12
П13
Р14
С15
Т16
У17"
trackable_list_wrapper
 "
trackable_list_wrapper
X
00
11
22
33
44
55
66
77"
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
Ф0
Х1"
trackable_list_wrapper
Є
Ъtrainable_variables
љmetrics
Њnon_trainable_variables
њlayers
Ыregularization_losses
 јlayer_regularization_losses
Ѕlayer_metrics
Ь	variables
∞__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
Ц0
Ч1
Ш2
Щ3"
trackable_list_wrapper
Є
Яtrainable_variables
¬metrics
√non_trainable_variables
ƒlayers
†regularization_losses
 ≈layer_regularization_losses
∆layer_metrics
°	variables
≤__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
£trainable_variables
«metrics
»non_trainable_variables
…layers
§regularization_losses
  layer_regularization_losses
Ћlayer_metrics
•	variables
і__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
P
Ф0
Х1
Ц2
Ч3
Ш4
Щ5"
trackable_list_wrapper
 "
trackable_list_wrapper
5
A0
B1
C2"
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
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
Є
±trainable_variables
ћmetrics
Ќnon_trainable_variables
ќlayers
≤regularization_losses
 ѕlayer_regularization_losses
–layer_metrics
≥	variables
ґ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
>
b0
c1
Ъ2
Ы3"
trackable_list_wrapper
Є
ґtrainable_variables
—metrics
“non_trainable_variables
”layers
Јregularization_losses
 ‘layer_regularization_losses
’layer_metrics
Є	variables
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Їtrainable_variables
÷metrics
„non_trainable_variables
Ўlayers
їregularization_losses
 ўlayer_regularization_losses
Џlayer_metrics
Љ	variables
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
0
Ъ0
Ы1"
trackable_list_wrapper
 "
trackable_list_wrapper
5
M0
N1
O2"
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
њ

џtotal

№count
Ё	variables
ё	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Б

яtotal

аcount
б
_fn_kwargs
в	variables
г	keras_api"µ
_tf_keras_metricЪ{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
f0
g1
h2
i3"
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
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
l0
m1
n2
o3"
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
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
r0
s1
t2
u3"
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
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
x0
y1
z2
{3"
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
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
>
~0
1
А2
Б3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
@
Д0
Е1
Ж2
З3"
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
0
И0
Й1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
@
К0
Л1
М2
Н3"
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
0
О0
П1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
@
Р0
С1
Т2
У3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Ф0
Х1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
@
Ц0
Ч1
Ш2
Щ3"
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
0
Ъ0
Ы1"
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
:  (2total
:  (2count
0
џ0
№1"
trackable_list_wrapper
.
Ё	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
я0
а1"
trackable_list_wrapper
.
в	variables"
_generic_user_object
/:-R2Adam/conv2d_11/kernel/m
!:2Adam/conv2d_11/bias/m
/:-2#Adam/batch_normalization_11/gamma/m
.:,2"Adam/batch_normalization_11/beta/m
/:-R2Adam/conv2d_11/kernel/v
!:2Adam/conv2d_11/bias/v
/:-2#Adam/batch_normalization_11/gamma/v
.:,2"Adam/batch_normalization_11/beta/v
и2е
#__inference__wrapped_model_34124916љ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *-Ґ*
(К%
input€€€€€€€€€ь
ц2у
*__inference_model_1_layer_call_fn_34129816
*__inference_model_1_layer_call_fn_34129691
*__inference_model_1_layer_call_fn_34128979
*__inference_model_1_layer_call_fn_34128716ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я
E__inference_model_1_layer_call_and_return_conditional_losses_34129566
E__inference_model_1_layer_call_and_return_conditional_losses_34128452
E__inference_model_1_layer_call_and_return_conditional_losses_34128314
E__inference_model_1_layer_call_and_return_conditional_losses_34129340ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
В2€
-__inference_input_conv_layer_call_fn_34129966
-__inference_input_conv_layer_call_fn_34125502
-__inference_input_conv_layer_call_fn_34129937
-__inference_input_conv_layer_call_fn_34125439ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
о2л
H__inference_input_conv_layer_call_and_return_conditional_losses_34129908
H__inference_input_conv_layer_call_and_return_conditional_losses_34125341
H__inference_input_conv_layer_call_and_return_conditional_losses_34129862
H__inference_input_conv_layer_call_and_return_conditional_losses_34125375ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
.__inference_res_block_0_layer_call_fn_34130180
.__inference_res_block_0_layer_call_fn_34130139
.__inference_res_block_0_layer_call_fn_34126351
.__inference_res_block_0_layer_call_fn_34126262ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
т2п
I__inference_res_block_0_layer_call_and_return_conditional_losses_34130098
I__inference_res_block_0_layer_call_and_return_conditional_losses_34130032
I__inference_res_block_0_layer_call_and_return_conditional_losses_34126172
I__inference_res_block_0_layer_call_and_return_conditional_losses_34126124ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
”2–
)__inference_relu_0_layer_call_fn_34130190Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_relu_0_layer_call_and_return_conditional_losses_34130185Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ж2Г
.__inference_res_block_1_layer_call_fn_34130404
.__inference_res_block_1_layer_call_fn_34130363
.__inference_res_block_1_layer_call_fn_34127111
.__inference_res_block_1_layer_call_fn_34127200ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
т2п
I__inference_res_block_1_layer_call_and_return_conditional_losses_34130256
I__inference_res_block_1_layer_call_and_return_conditional_losses_34126973
I__inference_res_block_1_layer_call_and_return_conditional_losses_34127021
I__inference_res_block_1_layer_call_and_return_conditional_losses_34130322ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
”2–
)__inference_relu_1_layer_call_fn_34130414Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_relu_1_layer_call_and_return_conditional_losses_34130409Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ц2у
*__inference_feature_layer_call_fn_34127471
*__inference_feature_layer_call_fn_34130498
*__inference_feature_layer_call_fn_34127507
*__inference_feature_layer_call_fn_34130481ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я
E__inference_feature_layer_call_and_return_conditional_losses_34130464
E__inference_feature_layer_call_and_return_conditional_losses_34130439
E__inference_feature_layer_call_and_return_conditional_losses_34127434
E__inference_feature_layer_call_and_return_conditional_losses_34127415ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
џ2Ў
1__inference_feature_linear_layer_call_fn_34130507Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ц2у
L__inference_feature_linear_layer_call_and_return_conditional_losses_34130502Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ж2Г
.__inference_output_conv_layer_call_fn_34130607
.__inference_output_conv_layer_call_fn_34130590
.__inference_output_conv_layer_call_fn_34127827
.__inference_output_conv_layer_call_fn_34127791ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
т2п
I__inference_output_conv_layer_call_and_return_conditional_losses_34127735
I__inference_output_conv_layer_call_and_return_conditional_losses_34130573
I__inference_output_conv_layer_call_and_return_conditional_losses_34130541
I__inference_output_conv_layer_call_and_return_conditional_losses_34127754ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Џ2„
0__inference_output_linear_layer_call_fn_34130616Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
х2т
K__inference_output_linear_layer_call_and_return_conditional_losses_34130611Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
‘2—
*__inference_softmax_layer_call_fn_34130626Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
п2м
E__inference_softmax_layer_call_and_return_conditional_losses_34130621Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЋB»
&__inference_signature_wrapper_34129112input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_conv2d_layer_call_fn_34130645Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_layer_call_and_return_conditional_losses_34130636Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ъ2Ч
6__inference_batch_normalization_layer_call_fn_34130694
6__inference_batch_normalization_layer_call_fn_34130756
6__inference_batch_normalization_layer_call_fn_34130707
6__inference_batch_normalization_layer_call_fn_34130769і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34130743
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34130681
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34130663
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34130725і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
„2‘
-__inference_activation_layer_call_fn_34130779Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_activation_layer_call_and_return_conditional_losses_34130774Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_conv2d_1_layer_call_fn_34130798Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_conv2d_1_layer_call_and_return_conditional_losses_34130789Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ґ2Я
8__inference_batch_normalization_1_layer_call_fn_34130922
8__inference_batch_normalization_1_layer_call_fn_34130860
8__inference_batch_normalization_1_layer_call_fn_34130847
8__inference_batch_normalization_1_layer_call_fn_34130909і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
О2Л
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34130816
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34130834
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34130878
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34130896і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ў2÷
/__inference_activation_1_layer_call_fn_34130932Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ф2с
J__inference_activation_1_layer_call_and_return_conditional_losses_34130927Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_conv2d_2_layer_call_fn_34130951Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_conv2d_2_layer_call_and_return_conditional_losses_34130942Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ґ2Я
8__inference_batch_normalization_2_layer_call_fn_34131075
8__inference_batch_normalization_2_layer_call_fn_34131013
8__inference_batch_normalization_2_layer_call_fn_34131000
8__inference_batch_normalization_2_layer_call_fn_34131062і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
О2Л
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34131049
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34130987
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34131031
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34130969і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ў2÷
/__inference_activation_2_layer_call_fn_34131085Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ф2с
J__inference_activation_2_layer_call_and_return_conditional_losses_34131080Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_conv2d_3_layer_call_fn_34131104Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_conv2d_3_layer_call_and_return_conditional_losses_34131095Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ґ2Я
8__inference_batch_normalization_3_layer_call_fn_34131228
8__inference_batch_normalization_3_layer_call_fn_34131153
8__inference_batch_normalization_3_layer_call_fn_34131166
8__inference_batch_normalization_3_layer_call_fn_34131215і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
О2Л
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34131122
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34131202
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34131140
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34131184і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ў2÷
/__inference_activation_3_layer_call_fn_34131238Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ф2с
J__inference_activation_3_layer_call_and_return_conditional_losses_34131233Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_conv2d_4_layer_call_fn_34131257Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_conv2d_4_layer_call_and_return_conditional_losses_34131248Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ґ2Я
8__inference_batch_normalization_4_layer_call_fn_34131368
8__inference_batch_normalization_4_layer_call_fn_34131381
8__inference_batch_normalization_4_layer_call_fn_34131306
8__inference_batch_normalization_4_layer_call_fn_34131319і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
О2Л
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34131337
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34131355
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34131293
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34131275і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
’2“
+__inference_conv2d_5_layer_call_fn_34131400Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_conv2d_5_layer_call_and_return_conditional_losses_34131391Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ґ2Я
8__inference_batch_normalization_5_layer_call_fn_34131524
8__inference_batch_normalization_5_layer_call_fn_34131462
8__inference_batch_normalization_5_layer_call_fn_34131449
8__inference_batch_normalization_5_layer_call_fn_34131511і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
О2Л
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34131436
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34131498
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34131418
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34131480і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ў2÷
/__inference_activation_4_layer_call_fn_34131534Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ф2с
J__inference_activation_4_layer_call_and_return_conditional_losses_34131529Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_conv2d_6_layer_call_fn_34131553Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_conv2d_6_layer_call_and_return_conditional_losses_34131544Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ґ2Я
8__inference_batch_normalization_6_layer_call_fn_34131602
8__inference_batch_normalization_6_layer_call_fn_34131677
8__inference_batch_normalization_6_layer_call_fn_34131664
8__inference_batch_normalization_6_layer_call_fn_34131615і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
О2Л
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34131571
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34131633
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34131589
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34131651і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ў2÷
/__inference_activation_5_layer_call_fn_34131687Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ф2с
J__inference_activation_5_layer_call_and_return_conditional_losses_34131682Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_conv2d_7_layer_call_fn_34131706Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_conv2d_7_layer_call_and_return_conditional_losses_34131697Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ґ2Я
8__inference_batch_normalization_7_layer_call_fn_34131817
8__inference_batch_normalization_7_layer_call_fn_34131755
8__inference_batch_normalization_7_layer_call_fn_34131768
8__inference_batch_normalization_7_layer_call_fn_34131830і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
О2Л
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34131786
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34131742
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34131804
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34131724і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
’2“
+__inference_conv2d_8_layer_call_fn_34131849Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_conv2d_8_layer_call_and_return_conditional_losses_34131840Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ґ2Я
8__inference_batch_normalization_8_layer_call_fn_34131973
8__inference_batch_normalization_8_layer_call_fn_34131911
8__inference_batch_normalization_8_layer_call_fn_34131960
8__inference_batch_normalization_8_layer_call_fn_34131898і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
О2Л
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34131929
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34131947
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34131885
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34131867і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ў2÷
/__inference_activation_6_layer_call_fn_34131983Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ф2с
J__inference_activation_6_layer_call_and_return_conditional_losses_34131978Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_conv2d_11_layer_call_fn_34132002Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_conv2d_11_layer_call_and_return_conditional_losses_34131993Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
¶2£
9__inference_batch_normalization_11_layer_call_fn_34132130
9__inference_batch_normalization_11_layer_call_fn_34132066
9__inference_batch_normalization_11_layer_call_fn_34132053
9__inference_batch_normalization_11_layer_call_fn_34132117і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Т2П
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34132086
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34132104
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34132040
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34132022і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
,__inference_reshape_1_layer_call_fn_34132147Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_reshape_1_layer_call_and_return_conditional_losses_34132142Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 о
#__inference__wrapped_model_34124916∆Xdefghijklmnopqrstuvwxyz{|}~АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩ`abcЪЫ7Ґ4
-Ґ*
(К%
input€€€€€€€€€ь
™ "1™.
,
softmax!К
softmax€€€€€€€€€ґ
J__inference_activation_1_layer_call_and_return_conditional_losses_34130927h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ О
/__inference_activation_1_layer_call_fn_34130932[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ " К€€€€€€€€€R@ґ
J__inference_activation_2_layer_call_and_return_conditional_losses_34131080h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ О
/__inference_activation_2_layer_call_fn_34131085[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ " К€€€€€€€€€R@ґ
J__inference_activation_3_layer_call_and_return_conditional_losses_34131233h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ О
/__inference_activation_3_layer_call_fn_34131238[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ " К€€€€€€€€€R@ґ
J__inference_activation_4_layer_call_and_return_conditional_losses_34131529h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ О
/__inference_activation_4_layer_call_fn_34131534[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ " К€€€€€€€€€R@ґ
J__inference_activation_5_layer_call_and_return_conditional_losses_34131682h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ О
/__inference_activation_5_layer_call_fn_34131687[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ " К€€€€€€€€€R@ґ
J__inference_activation_6_layer_call_and_return_conditional_losses_34131978h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R
™ "-Ґ*
#К 
0€€€€€€€€€R
Ъ О
/__inference_activation_6_layer_call_fn_34131983[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R
™ " К€€€€€€€€€RЄ
H__inference_activation_layer_call_and_return_conditional_losses_34130774l9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€®А
™ "/Ґ,
%К"
0€€€€€€€€€®А
Ъ Р
-__inference_activation_layer_call_fn_34130779_9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€®А
™ ""К€€€€€€€€€®Аћ
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34132022tbcЪЫ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ ћ
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34132040tbcЪЫ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ с
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34132086ШbcЪЫMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ с
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_34132104ШbcЪЫMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ §
9__inference_batch_normalization_11_layer_call_fn_34132053gbcЪЫ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p
™ " К€€€€€€€€€§
9__inference_batch_normalization_11_layer_call_fn_34132066gbcЪЫ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€
p 
™ " К€€€€€€€€€…
9__inference_batch_normalization_11_layer_call_fn_34132117ЛbcЪЫMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€…
9__inference_batch_normalization_11_layer_call_fn_34132130ЛbcЪЫMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€о
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34130816ЦlmnoMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ о
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34130834ЦlmnoMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ …
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34130878rlmno;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ …
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_34130896rlmno;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ ∆
8__inference_batch_normalization_1_layer_call_fn_34130847ЙlmnoMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∆
8__inference_batch_normalization_1_layer_call_fn_34130860ЙlmnoMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@°
8__inference_batch_normalization_1_layer_call_fn_34130909elmno;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p
™ " К€€€€€€€€€R@°
8__inference_batch_normalization_1_layer_call_fn_34130922elmno;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p 
™ " К€€€€€€€€€R@…
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34130969rrstu;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ …
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34130987rrstu;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ о
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34131031ЦrstuMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ о
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_34131049ЦrstuMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ °
8__inference_batch_normalization_2_layer_call_fn_34131000erstu;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p
™ " К€€€€€€€€€R@°
8__inference_batch_normalization_2_layer_call_fn_34131013erstu;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p 
™ " К€€€€€€€€€R@∆
8__inference_batch_normalization_2_layer_call_fn_34131062ЙrstuMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∆
8__inference_batch_normalization_2_layer_call_fn_34131075ЙrstuMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@о
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34131122Цxyz{MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ о
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34131140Цxyz{MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ …
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34131184rxyz{;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ …
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_34131202rxyz{;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ ∆
8__inference_batch_normalization_3_layer_call_fn_34131153Йxyz{MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∆
8__inference_batch_normalization_3_layer_call_fn_34131166Йxyz{MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@°
8__inference_batch_normalization_3_layer_call_fn_34131215exyz{;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p
™ " К€€€€€€€€€R@°
8__inference_batch_normalization_3_layer_call_fn_34131228exyz{;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p 
™ " К€€€€€€€€€R@р
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34131275Ш~АБMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ р
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34131293Ш~АБMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ Ћ
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34131337t~АБ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ Ћ
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_34131355t~АБ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ »
8__inference_batch_normalization_4_layer_call_fn_34131306Л~АБMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@»
8__inference_batch_normalization_4_layer_call_fn_34131319Л~АБMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@£
8__inference_batch_normalization_4_layer_call_fn_34131368g~АБ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p
™ " К€€€€€€€€€R@£
8__inference_batch_normalization_4_layer_call_fn_34131381g~АБ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p 
™ " К€€€€€€€€€R@т
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34131418ЪДЕЖЗMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ т
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34131436ЪДЕЖЗMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ Ќ
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34131480vДЕЖЗ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ Ќ
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_34131498vДЕЖЗ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ  
8__inference_batch_normalization_5_layer_call_fn_34131449НДЕЖЗMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ 
8__inference_batch_normalization_5_layer_call_fn_34131462НДЕЖЗMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@•
8__inference_batch_normalization_5_layer_call_fn_34131511iДЕЖЗ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p
™ " К€€€€€€€€€R@•
8__inference_batch_normalization_5_layer_call_fn_34131524iДЕЖЗ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p 
™ " К€€€€€€€€€R@т
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34131571ЪКЛМНMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ т
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34131589ЪКЛМНMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ Ќ
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34131633vКЛМН;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ Ќ
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_34131651vКЛМН;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ  
8__inference_batch_normalization_6_layer_call_fn_34131602НКЛМНMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ 
8__inference_batch_normalization_6_layer_call_fn_34131615НКЛМНMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@•
8__inference_batch_normalization_6_layer_call_fn_34131664iКЛМН;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p
™ " К€€€€€€€€€R@•
8__inference_batch_normalization_6_layer_call_fn_34131677iКЛМН;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p 
™ " К€€€€€€€€€R@Ќ
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34131724vРСТУ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ Ќ
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34131742vРСТУ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ т
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34131786ЪРСТУMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ т
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_34131804ЪРСТУMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ •
8__inference_batch_normalization_7_layer_call_fn_34131755iРСТУ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p
™ " К€€€€€€€€€R@•
8__inference_batch_normalization_7_layer_call_fn_34131768iРСТУ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R@
p 
™ " К€€€€€€€€€R@ 
8__inference_batch_normalization_7_layer_call_fn_34131817НРСТУMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ 
8__inference_batch_normalization_7_layer_call_fn_34131830НРСТУMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ќ
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34131867vЦЧШЩ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R
p
™ "-Ґ*
#К 
0€€€€€€€€€R
Ъ Ќ
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34131885vЦЧШЩ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R
p 
™ "-Ґ*
#К 
0€€€€€€€€€R
Ъ т
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34131929ЪЦЧШЩMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ т
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_34131947ЪЦЧШЩMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ •
8__inference_batch_normalization_8_layer_call_fn_34131898iЦЧШЩ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R
p
™ " К€€€€€€€€€R•
8__inference_batch_normalization_8_layer_call_fn_34131911iЦЧШЩ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€R
p 
™ " К€€€€€€€€€R 
8__inference_batch_normalization_8_layer_call_fn_34131960НЦЧШЩMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
8__inference_batch_normalization_8_layer_call_fn_34131973НЦЧШЩMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€о
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34130663ШfghiNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ о
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34130681ШfghiNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Ћ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34130725vfghi=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€®А
p
™ "/Ґ,
%К"
0€€€€€€€€€®А
Ъ Ћ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_34130743vfghi=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€®А
p 
™ "/Ґ,
%К"
0€€€€€€€€€®А
Ъ ∆
6__inference_batch_normalization_layer_call_fn_34130694ЛfghiNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А∆
6__inference_batch_normalization_layer_call_fn_34130707ЛfghiNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А£
6__inference_batch_normalization_layer_call_fn_34130756ifghi=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€®А
p
™ ""К€€€€€€€€€®А£
6__inference_batch_normalization_layer_call_fn_34130769ifghi=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€®А
p 
™ ""К€€€€€€€€€®АЈ
G__inference_conv2d_11_layer_call_and_return_conditional_losses_34131993l`a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ П
,__inference_conv2d_11_layer_call_fn_34132002_`a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R
™ " К€€€€€€€€€Є
F__inference_conv2d_1_layer_call_and_return_conditional_losses_34130789njk9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€®А
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ Р
+__inference_conv2d_1_layer_call_fn_34130798ajk9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€®А
™ " К€€€€€€€€€R@ґ
F__inference_conv2d_2_layer_call_and_return_conditional_losses_34130942lpq7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ О
+__inference_conv2d_2_layer_call_fn_34130951_pq7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ " К€€€€€€€€€R@ґ
F__inference_conv2d_3_layer_call_and_return_conditional_losses_34131095lvw7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ О
+__inference_conv2d_3_layer_call_fn_34131104_vw7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ " К€€€€€€€€€R@ґ
F__inference_conv2d_4_layer_call_and_return_conditional_losses_34131248l|}7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ О
+__inference_conv2d_4_layer_call_fn_34131257_|}7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ " К€€€€€€€€€R@Є
F__inference_conv2d_5_layer_call_and_return_conditional_losses_34131391nВГ7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ Р
+__inference_conv2d_5_layer_call_fn_34131400aВГ7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ " К€€€€€€€€€R@Є
F__inference_conv2d_6_layer_call_and_return_conditional_losses_34131544nИЙ7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ Р
+__inference_conv2d_6_layer_call_fn_34131553aИЙ7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ " К€€€€€€€€€R@Є
F__inference_conv2d_7_layer_call_and_return_conditional_losses_34131697nОП7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ Р
+__inference_conv2d_7_layer_call_fn_34131706aОП7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ " К€€€€€€€€€R@Є
F__inference_conv2d_8_layer_call_and_return_conditional_losses_34131840nФХ7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ "-Ґ*
#К 
0€€€€€€€€€R
Ъ Р
+__inference_conv2d_8_layer_call_fn_34131849aФХ7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ " К€€€€€€€€€RЈ
D__inference_conv2d_layer_call_and_return_conditional_losses_34130636ode8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€ь
™ "/Ґ,
%К"
0€€€€€€€€€®А
Ъ П
)__inference_conv2d_layer_call_fn_34130645bde8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€ь
™ ""К€€€€€€€€€®А–
E__inference_feature_layer_call_and_return_conditional_losses_34127415ЖФХЦЧШЩGҐD
=Ґ:
0К-
conv2d_8_input€€€€€€€€€R@
p

 
™ "-Ґ*
#К 
0€€€€€€€€€R
Ъ –
E__inference_feature_layer_call_and_return_conditional_losses_34127434ЖФХЦЧШЩGҐD
=Ґ:
0К-
conv2d_8_input€€€€€€€€€R@
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€R
Ъ «
E__inference_feature_layer_call_and_return_conditional_losses_34130439~ФХЦЧШЩ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R@
p

 
™ "-Ґ*
#К 
0€€€€€€€€€R
Ъ «
E__inference_feature_layer_call_and_return_conditional_losses_34130464~ФХЦЧШЩ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R@
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€R
Ъ І
*__inference_feature_layer_call_fn_34127471yФХЦЧШЩGҐD
=Ґ:
0К-
conv2d_8_input€€€€€€€€€R@
p

 
™ " К€€€€€€€€€RІ
*__inference_feature_layer_call_fn_34127507yФХЦЧШЩGҐD
=Ґ:
0К-
conv2d_8_input€€€€€€€€€R@
p 

 
™ " К€€€€€€€€€RЯ
*__inference_feature_layer_call_fn_34130481qФХЦЧШЩ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R@
p

 
™ " К€€€€€€€€€RЯ
*__inference_feature_layer_call_fn_34130498qФХЦЧШЩ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R@
p 

 
™ " К€€€€€€€€€RЄ
L__inference_feature_linear_layer_call_and_return_conditional_losses_34130502h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R
™ "-Ґ*
#К 
0€€€€€€€€€R
Ъ Р
1__inference_feature_linear_layer_call_fn_34130507[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R
™ " К€€€€€€€€€R“
H__inference_input_conv_layer_call_and_return_conditional_losses_34125341ЕdefghijklmnoFҐC
<Ґ9
/К,
conv2d_input€€€€€€€€€ь
p

 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ “
H__inference_input_conv_layer_call_and_return_conditional_losses_34125375ЕdefghijklmnoFҐC
<Ґ9
/К,
conv2d_input€€€€€€€€€ь
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ Ћ
H__inference_input_conv_layer_call_and_return_conditional_losses_34129862defghijklmno@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€ь
p

 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ Ћ
H__inference_input_conv_layer_call_and_return_conditional_losses_34129908defghijklmno@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€ь
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ ©
-__inference_input_conv_layer_call_fn_34125439xdefghijklmnoFҐC
<Ґ9
/К,
conv2d_input€€€€€€€€€ь
p

 
™ " К€€€€€€€€€R@©
-__inference_input_conv_layer_call_fn_34125502xdefghijklmnoFҐC
<Ґ9
/К,
conv2d_input€€€€€€€€€ь
p 

 
™ " К€€€€€€€€€R@£
-__inference_input_conv_layer_call_fn_34129937rdefghijklmno@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€ь
p

 
™ " К€€€€€€€€€R@£
-__inference_input_conv_layer_call_fn_34129966rdefghijklmno@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€ь
p 

 
™ " К€€€€€€€€€R@М
E__inference_model_1_layer_call_and_return_conditional_losses_34128314¬Xdefghijklmnopqrstuvwxyz{|}~АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩ`abcЪЫ?Ґ<
5Ґ2
(К%
input€€€€€€€€€ь
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ М
E__inference_model_1_layer_call_and_return_conditional_losses_34128452¬Xdefghijklmnopqrstuvwxyz{|}~АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩ`abcЪЫ?Ґ<
5Ґ2
(К%
input€€€€€€€€€ь
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Н
E__inference_model_1_layer_call_and_return_conditional_losses_34129340√Xdefghijklmnopqrstuvwxyz{|}~АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩ`abcЪЫ@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€ь
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Н
E__inference_model_1_layer_call_and_return_conditional_losses_34129566√Xdefghijklmnopqrstuvwxyz{|}~АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩ`abcЪЫ@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€ь
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ д
*__inference_model_1_layer_call_fn_34128716µXdefghijklmnopqrstuvwxyz{|}~АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩ`abcЪЫ?Ґ<
5Ґ2
(К%
input€€€€€€€€€ь
p

 
™ "К€€€€€€€€€д
*__inference_model_1_layer_call_fn_34128979µXdefghijklmnopqrstuvwxyz{|}~АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩ`abcЪЫ?Ґ<
5Ґ2
(К%
input€€€€€€€€€ь
p 

 
™ "К€€€€€€€€€е
*__inference_model_1_layer_call_fn_34129691ґXdefghijklmnopqrstuvwxyz{|}~АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩ`abcЪЫ@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€ь
p

 
™ "К€€€€€€€€€е
*__inference_model_1_layer_call_fn_34129816ґXdefghijklmnopqrstuvwxyz{|}~АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩ`abcЪЫ@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€ь
p 

 
™ "К€€€€€€€€€»
I__inference_output_conv_layer_call_and_return_conditional_losses_34127735{`abcЪЫHҐE
>Ґ;
1К.
conv2d_11_input€€€€€€€€€R
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ »
I__inference_output_conv_layer_call_and_return_conditional_losses_34127754{`abcЪЫHҐE
>Ґ;
1К.
conv2d_11_input€€€€€€€€€R
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ њ
I__inference_output_conv_layer_call_and_return_conditional_losses_34130541r`abcЪЫ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ њ
I__inference_output_conv_layer_call_and_return_conditional_losses_34130573r`abcЪЫ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ †
.__inference_output_conv_layer_call_fn_34127791n`abcЪЫHҐE
>Ґ;
1К.
conv2d_11_input€€€€€€€€€R
p

 
™ "К€€€€€€€€€†
.__inference_output_conv_layer_call_fn_34127827n`abcЪЫHҐE
>Ґ;
1К.
conv2d_11_input€€€€€€€€€R
p 

 
™ "К€€€€€€€€€Ч
.__inference_output_conv_layer_call_fn_34130590e`abcЪЫ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R
p

 
™ "К€€€€€€€€€Ч
.__inference_output_conv_layer_call_fn_34130607e`abcЪЫ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R
p 

 
™ "К€€€€€€€€€І
K__inference_output_linear_layer_call_and_return_conditional_losses_34130611X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ 
0__inference_output_linear_layer_call_fn_34130616K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€∞
D__inference_relu_0_layer_call_and_return_conditional_losses_34130185h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ И
)__inference_relu_0_layer_call_fn_34130190[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ " К€€€€€€€€€R@∞
D__inference_relu_1_layer_call_and_return_conditional_losses_34130409h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ И
)__inference_relu_1_layer_call_fn_34130414[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€R@
™ " К€€€€€€€€€R@№
I__inference_res_block_0_layer_call_and_return_conditional_losses_34126124Оpqrstuvwxyz{|}~АБGҐD
=Ґ:
0К-
conv2d_2_input€€€€€€€€€R@
p

 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ №
I__inference_res_block_0_layer_call_and_return_conditional_losses_34126172Оpqrstuvwxyz{|}~АБGҐD
=Ґ:
0К-
conv2d_2_input€€€€€€€€€R@
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ ‘
I__inference_res_block_0_layer_call_and_return_conditional_losses_34130032Жpqrstuvwxyz{|}~АБ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R@
p

 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ ‘
I__inference_res_block_0_layer_call_and_return_conditional_losses_34130098Жpqrstuvwxyz{|}~АБ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R@
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ і
.__inference_res_block_0_layer_call_fn_34126262Бpqrstuvwxyz{|}~АБGҐD
=Ґ:
0К-
conv2d_2_input€€€€€€€€€R@
p

 
™ " К€€€€€€€€€R@і
.__inference_res_block_0_layer_call_fn_34126351Бpqrstuvwxyz{|}~АБGҐD
=Ґ:
0К-
conv2d_2_input€€€€€€€€€R@
p 

 
™ " К€€€€€€€€€R@Ђ
.__inference_res_block_0_layer_call_fn_34130139ypqrstuvwxyz{|}~АБ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R@
p

 
™ " К€€€€€€€€€R@Ђ
.__inference_res_block_0_layer_call_fn_34130180ypqrstuvwxyz{|}~АБ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R@
p 

 
™ " К€€€€€€€€€R@м
I__inference_res_block_1_layer_call_and_return_conditional_losses_34126973Ю$ВГДЕЖЗИЙКЛМНОПРСТУGҐD
=Ґ:
0К-
conv2d_5_input€€€€€€€€€R@
p

 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ м
I__inference_res_block_1_layer_call_and_return_conditional_losses_34127021Ю$ВГДЕЖЗИЙКЛМНОПРСТУGҐD
=Ґ:
0К-
conv2d_5_input€€€€€€€€€R@
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ д
I__inference_res_block_1_layer_call_and_return_conditional_losses_34130256Ц$ВГДЕЖЗИЙКЛМНОПРСТУ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R@
p

 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ д
I__inference_res_block_1_layer_call_and_return_conditional_losses_34130322Ц$ВГДЕЖЗИЙКЛМНОПРСТУ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R@
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€R@
Ъ ƒ
.__inference_res_block_1_layer_call_fn_34127111С$ВГДЕЖЗИЙКЛМНОПРСТУGҐD
=Ґ:
0К-
conv2d_5_input€€€€€€€€€R@
p

 
™ " К€€€€€€€€€R@ƒ
.__inference_res_block_1_layer_call_fn_34127200С$ВГДЕЖЗИЙКЛМНОПРСТУGҐD
=Ґ:
0К-
conv2d_5_input€€€€€€€€€R@
p 

 
™ " К€€€€€€€€€R@Љ
.__inference_res_block_1_layer_call_fn_34130363Й$ВГДЕЖЗИЙКЛМНОПРСТУ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R@
p

 
™ " К€€€€€€€€€R@Љ
.__inference_res_block_1_layer_call_fn_34130404Й$ВГДЕЖЗИЙКЛМНОПРСТУ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€R@
p 

 
™ " К€€€€€€€€€R@Ђ
G__inference_reshape_1_layer_call_and_return_conditional_losses_34132142`7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ Г
,__inference_reshape_1_layer_call_fn_34132147S7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "К€€€€€€€€€ъ
&__inference_signature_wrapper_34129112ѕXdefghijklmnopqrstuvwxyz{|}~АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩ`abcЪЫ@Ґ=
Ґ 
6™3
1
input(К%
input€€€€€€€€€ь"1™.
,
softmax!К
softmax€€€€€€€€€°
E__inference_softmax_layer_call_and_return_conditional_losses_34130621X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ y
*__inference_softmax_layer_call_fn_34130626K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€