��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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
delete_old_dirsbool(�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68�
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:d*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
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
�
%simple_rnn_4/simple_rnn_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*6
shared_name'%simple_rnn_4/simple_rnn_cell_4/kernel
�
9simple_rnn_4/simple_rnn_cell_4/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_4/simple_rnn_cell_4/kernel*
_output_shapes

:d*
dtype0
�
/simple_rnn_4/simple_rnn_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*@
shared_name1/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel
�
Csimple_rnn_4/simple_rnn_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel*
_output_shapes

:dd*
dtype0
�
#simple_rnn_4/simple_rnn_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#simple_rnn_4/simple_rnn_cell_4/bias
�
7simple_rnn_4/simple_rnn_cell_4/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_4/simple_rnn_cell_4/bias*
_output_shapes
:d*
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
�
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0
�
,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*=
shared_name.,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/m
�
@Adam/simple_rnn_4/simple_rnn_cell_4/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/m*
_output_shapes

:d*
dtype0
�
6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*G
shared_name86Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/m
�
JAdam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/m*
_output_shapes

:dd*
dtype0
�
*Adam/simple_rnn_4/simple_rnn_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*;
shared_name,*Adam/simple_rnn_4/simple_rnn_cell_4/bias/m
�
>Adam/simple_rnn_4/simple_rnn_cell_4/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_4/simple_rnn_cell_4/bias/m*
_output_shapes
:d*
dtype0
�
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0
�
,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*=
shared_name.,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/v
�
@Adam/simple_rnn_4/simple_rnn_cell_4/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/v*
_output_shapes

:d*
dtype0
�
6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*G
shared_name86Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/v
�
JAdam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/v*
_output_shapes

:dd*
dtype0
�
*Adam/simple_rnn_4/simple_rnn_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*;
shared_name,*Adam/simple_rnn_4/simple_rnn_cell_4/bias/v
�
>Adam/simple_rnn_4/simple_rnn_cell_4/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_4/simple_rnn_cell_4/bias/v*
_output_shapes
:d*
dtype0

NoOpNoOp
�(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�(
value�(B�( B�(
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures*
�
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�
iter

beta_1

beta_2
	decay
 learning_ratemFmG!mH"mI#mJvKvL!vM"vN#vO*
'
!0
"1
#2
3
4*
'
!0
"1
#2
3
4*
* 
�
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
* 
* 
* 

)serving_default* 
�

!kernel
"recurrent_kernel
#bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
._random_generator
/__call__
*0&call_and_return_all_conditional_losses*
* 

!0
"1
#2*

!0
"1
#2*
* 
�

1states
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_4/simple_rnn_cell_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_4/simple_rnn_cell_4/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

<0*
* 
* 
* 

!0
"1
#2*

!0
"1
#2*
* 
�
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
*	variables
+trainable_variables
,regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
8
	Btotal
	Ccount
D	variables
E	keras_api*
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

B0
C1*

D	variables*
�{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/simple_rnn_4/simple_rnn_cell_4/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/simple_rnn_4/simple_rnn_cell_4/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
"serving_default_simple_rnn_4_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall"serving_default_simple_rnn_4_input%simple_rnn_4/simple_rnn_cell_4/kernel#simple_rnn_4/simple_rnn_cell_4/bias/simple_rnn_4/simple_rnn_cell_4/recurrent_kerneldense_4/kerneldense_4/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_203518
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9simple_rnn_4/simple_rnn_cell_4/kernel/Read/ReadVariableOpCsimple_rnn_4/simple_rnn_cell_4/recurrent_kernel/Read/ReadVariableOp7simple_rnn_4/simple_rnn_cell_4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp@Adam/simple_rnn_4/simple_rnn_cell_4/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_4/simple_rnn_cell_4/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp@Adam/simple_rnn_4/simple_rnn_cell_4/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_4/simple_rnn_cell_4/bias/v/Read/ReadVariableOpConst*#
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_204164
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%simple_rnn_4/simple_rnn_cell_4/kernel/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel#simple_rnn_4/simple_rnn_cell_4/biastotalcountAdam/dense_4/kernel/mAdam/dense_4/bias/m,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/m6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/m*Adam/simple_rnn_4/simple_rnn_cell_4/bias/mAdam/dense_4/kernel/vAdam/dense_4/bias/v,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/v6Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/v*Adam/simple_rnn_4/simple_rnn_cell_4/bias/v*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_204240��
�
�
-__inference_simple_rnn_4_layer_call_fn_203529
inputs_0
unknown:d
	unknown_0:d
	unknown_1:dd
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_202684o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�

�
2__inference_simple_rnn_cell_4_layer_call_fn_204041

inputs
states_0
unknown:d
	unknown_0:d
	unknown_1:dd
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_202728o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0
�
�
while_cond_203819
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_203819___redundant_placeholder04
0while_while_cond_203819___redundant_placeholder14
0while_while_cond_203819___redundant_placeholder24
0while_while_cond_203819___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�

�
simple_rnn_4_while_cond_2033146
2simple_rnn_4_while_simple_rnn_4_while_loop_counter<
8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations"
simple_rnn_4_while_placeholder$
 simple_rnn_4_while_placeholder_1$
 simple_rnn_4_while_placeholder_28
4simple_rnn_4_while_less_simple_rnn_4_strided_slice_1N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_203314___redundant_placeholder0N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_203314___redundant_placeholder1N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_203314___redundant_placeholder2N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_203314___redundant_placeholder3
simple_rnn_4_while_identity
�
simple_rnn_4/while/LessLesssimple_rnn_4_while_placeholder4simple_rnn_4_while_less_simple_rnn_4_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_4/while/IdentityIdentitysimple_rnn_4/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_4_while_identity$simple_rnn_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�,
�
while_body_203069
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0:dG
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:dL
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_4_matmul_readvariableop_resource:dE
7while_simple_rnn_cell_4_biasadd_readvariableop_resource:dJ
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:dd��.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_4/MatMul/ReadVariableOp�/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0�
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0�
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������dw
while/simple_rnn_cell_4/ReluReluwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_4/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity*while/simple_rnn_cell_4/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_simple_rnn_4_layer_call_fn_203540
inputs_0
unknown:d
	unknown_0:d
	unknown_1:dd
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_202843o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�,
�
while_body_202900
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0:dG
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:dL
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_4_matmul_readvariableop_resource:dE
7while_simple_rnn_cell_4_biasadd_readvariableop_resource:dJ
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:dd��.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_4/MatMul/ReadVariableOp�/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0�
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0�
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������dw
while/simple_rnn_cell_4/ReluReluwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_4/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity*while/simple_rnn_cell_4/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_simple_rnn_4_layer_call_fn_203562

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203135o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dense_4_layer_call_fn_204003

inputs
unknown:d
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_202984o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�8
�
simple_rnn_4_while_body_2033156
2simple_rnn_4_while_simple_rnn_4_while_loop_counter<
8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations"
simple_rnn_4_while_placeholder$
 simple_rnn_4_while_placeholder_1$
 simple_rnn_4_while_placeholder_25
1simple_rnn_4_while_simple_rnn_4_strided_slice_1_0q
msimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0:dT
Fsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:dY
Gsimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:dd
simple_rnn_4_while_identity!
simple_rnn_4_while_identity_1!
simple_rnn_4_while_identity_2!
simple_rnn_4_while_identity_3!
simple_rnn_4_while_identity_43
/simple_rnn_4_while_simple_rnn_4_strided_slice_1o
ksimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource:dR
Dsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource:dW
Esimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource:dd��;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp�<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
Dsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
6simple_rnn_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_4_while_placeholderMsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
+simple_rnn_4/while/simple_rnn_cell_4/MatMulMatMul=simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0�
,simple_rnn_4/while/simple_rnn_cell_4/BiasAddBiasAdd5simple_rnn_4/while/simple_rnn_cell_4/MatMul:product:0Csimple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0�
-simple_rnn_4/while/simple_rnn_cell_4/MatMul_1MatMul simple_rnn_4_while_placeholder_2Dsimple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
(simple_rnn_4/while/simple_rnn_cell_4/addAddV25simple_rnn_4/while/simple_rnn_cell_4/BiasAdd:output:07simple_rnn_4/while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������d�
)simple_rnn_4/while/simple_rnn_cell_4/ReluRelu,simple_rnn_4/while/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������d�
7simple_rnn_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_4_while_placeholder_1simple_rnn_4_while_placeholder7simple_rnn_4/while/simple_rnn_cell_4/Relu:activations:0*
_output_shapes
: *
element_dtype0:���Z
simple_rnn_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_4/while/addAddV2simple_rnn_4_while_placeholder!simple_rnn_4/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_4/while/add_1AddV22simple_rnn_4_while_simple_rnn_4_while_loop_counter#simple_rnn_4/while/add_1/y:output:0*
T0*
_output_shapes
: �
simple_rnn_4/while/IdentityIdentitysimple_rnn_4/while/add_1:z:0^simple_rnn_4/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_4/while/Identity_1Identity8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations^simple_rnn_4/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_4/while/Identity_2Identitysimple_rnn_4/while/add:z:0^simple_rnn_4/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_4/while/Identity_3IdentityGsimple_rnn_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_4/while/NoOp*
T0*
_output_shapes
: :����
simple_rnn_4/while/Identity_4Identity7simple_rnn_4/while/simple_rnn_cell_4/Relu:activations:0^simple_rnn_4/while/NoOp*
T0*'
_output_shapes
:���������d�
simple_rnn_4/while/NoOpNoOp<^simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_4_while_identity$simple_rnn_4/while/Identity:output:0"G
simple_rnn_4_while_identity_1&simple_rnn_4/while/Identity_1:output:0"G
simple_rnn_4_while_identity_2&simple_rnn_4/while/Identity_2:output:0"G
simple_rnn_4_while_identity_3&simple_rnn_4/while/Identity_3:output:0"G
simple_rnn_4_while_identity_4&simple_rnn_4/while/Identity_4:output:0"d
/simple_rnn_4_while_simple_rnn_4_strided_slice_11simple_rnn_4_while_simple_rnn_4_strided_slice_1_0"�
Dsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resourceFsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"�
Esimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resourceGsimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"�
Csimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resourceEsimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0"�
ksimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensormsimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2z
;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2x
:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp2|
<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�6
�
__inference__traced_save_204164
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_simple_rnn_4_simple_rnn_cell_4_kernel_read_readvariableopN
Jsavev2_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_4_simple_rnn_cell_4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_4_simple_rnn_cell_4_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_4_simple_rnn_cell_4_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_4_simple_rnn_cell_4_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_4_simple_rnn_cell_4_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_simple_rnn_4_simple_rnn_cell_4_kernel_read_readvariableopJsavev2_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_read_readvariableop>savev2_simple_rnn_4_simple_rnn_cell_4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableopGsavev2_adam_simple_rnn_4_simple_rnn_cell_4_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_4_simple_rnn_cell_4_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableopGsavev2_adam_simple_rnn_4_simple_rnn_cell_4_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_4_simple_rnn_cell_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :d:: : : : : :d:dd:d: : :d::d:dd:d:d::d:dd:d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d:$	 

_output_shapes

:dd: 


_output_shapes
:d:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:

_output_shapes
: 
�=
�
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203886

inputsB
0simple_rnn_cell_4_matmul_readvariableop_resource:d?
1simple_rnn_cell_4_biasadd_readvariableop_resource:dD
2simple_rnn_cell_4_matmul_1_readvariableop_resource:dd
identity��(simple_rnn_cell_4/BiasAdd/ReadVariableOp�'simple_rnn_cell_4/MatMul/ReadVariableOp�)simple_rnn_cell_4/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0�
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������dk
simple_rnn_cell_4/ReluRelusimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_203820*
condR
while_cond_203819*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
C__inference_dense_4_layer_call_and_return_conditional_losses_204013

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�,
�
while_body_203820
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0:dG
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:dL
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_4_matmul_readvariableop_resource:dE
7while_simple_rnn_cell_4_biasadd_readvariableop_resource:dJ
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:dd��.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_4/MatMul/ReadVariableOp�/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0�
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0�
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������dw
while/simple_rnn_cell_4/ReluReluwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_4/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity*while/simple_rnn_cell_4/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_204058

inputs
states_00
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������dG
ReluReluadd:z:0*
T0*'
_output_shapes
:���������da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������dc

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0
�!
�
while_body_202621
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_4_202643_0:d.
 while_simple_rnn_cell_4_202645_0:d2
 while_simple_rnn_cell_4_202647_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_4_202643:d,
while_simple_rnn_cell_4_202645:d0
while_simple_rnn_cell_4_202647:dd��/while/simple_rnn_cell_4/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
/while/simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_4_202643_0 while_simple_rnn_cell_4_202645_0 while_simple_rnn_cell_4_202647_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_202608�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity8while/simple_rnn_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������d~

while/NoOpNoOp0^while/simple_rnn_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"B
while_simple_rnn_cell_4_202643 while_simple_rnn_cell_4_202643_0"B
while_simple_rnn_cell_4_202645 while_simple_rnn_cell_4_202645_0"B
while_simple_rnn_cell_4_202647 while_simple_rnn_cell_4_202647_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2b
/while/simple_rnn_cell_4/StatefulPartitionedCall/while/simple_rnn_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_203068
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_203068___redundant_placeholder04
0while_while_cond_203068___redundant_placeholder14
0while_while_cond_203068___redundant_placeholder24
0while_while_cond_203068___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_202779
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_202779___redundant_placeholder04
0while_while_cond_202779___redundant_placeholder14
0while_while_cond_202779___redundant_placeholder24
0while_while_cond_202779___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�

�
simple_rnn_4_while_cond_2034286
2simple_rnn_4_while_simple_rnn_4_while_loop_counter<
8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations"
simple_rnn_4_while_placeholder$
 simple_rnn_4_while_placeholder_1$
 simple_rnn_4_while_placeholder_28
4simple_rnn_4_while_less_simple_rnn_4_strided_slice_1N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_203428___redundant_placeholder0N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_203428___redundant_placeholder1N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_203428___redundant_placeholder2N
Jsimple_rnn_4_while_simple_rnn_4_while_cond_203428___redundant_placeholder3
simple_rnn_4_while_identity
�
simple_rnn_4/while/LessLesssimple_rnn_4_while_placeholder4simple_rnn_4_while_less_simple_rnn_4_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_4/while/IdentityIdentitysimple_rnn_4/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_4_while_identity$simple_rnn_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�=
�
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_202966

inputsB
0simple_rnn_cell_4_matmul_readvariableop_resource:d?
1simple_rnn_cell_4_biasadd_readvariableop_resource:dD
2simple_rnn_cell_4_matmul_1_readvariableop_resource:dd
identity��(simple_rnn_cell_4/BiasAdd/ReadVariableOp�'simple_rnn_cell_4/MatMul/ReadVariableOp�)simple_rnn_cell_4/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0�
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������dk
simple_rnn_cell_4/ReluRelusimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_202900*
condR
while_cond_202899*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_203518
simple_rnn_4_input
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_202560o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:���������
,
_user_specified_namesimple_rnn_4_input
�_
�
!__inference__wrapped_model_202560
simple_rnn_4_input\
Jsequential_4_simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resource:dY
Ksequential_4_simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resource:d^
Lsequential_4_simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource:ddE
3sequential_4_dense_4_matmul_readvariableop_resource:dB
4sequential_4_dense_4_biasadd_readvariableop_resource:
identity��+sequential_4/dense_4/BiasAdd/ReadVariableOp�*sequential_4/dense_4/MatMul/ReadVariableOp�Bsequential_4/simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp�Asequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp�Csequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp�sequential_4/simple_rnn_4/whilea
sequential_4/simple_rnn_4/ShapeShapesimple_rnn_4_input*
T0*
_output_shapes
:w
-sequential_4/simple_rnn_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential_4/simple_rnn_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential_4/simple_rnn_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'sequential_4/simple_rnn_4/strided_sliceStridedSlice(sequential_4/simple_rnn_4/Shape:output:06sequential_4/simple_rnn_4/strided_slice/stack:output:08sequential_4/simple_rnn_4/strided_slice/stack_1:output:08sequential_4/simple_rnn_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_4/simple_rnn_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
&sequential_4/simple_rnn_4/zeros/packedPack0sequential_4/simple_rnn_4/strided_slice:output:01sequential_4/simple_rnn_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%sequential_4/simple_rnn_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_4/simple_rnn_4/zerosFill/sequential_4/simple_rnn_4/zeros/packed:output:0.sequential_4/simple_rnn_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������d}
(sequential_4/simple_rnn_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
#sequential_4/simple_rnn_4/transpose	Transposesimple_rnn_4_input1sequential_4/simple_rnn_4/transpose/perm:output:0*
T0*+
_output_shapes
:���������x
!sequential_4/simple_rnn_4/Shape_1Shape'sequential_4/simple_rnn_4/transpose:y:0*
T0*
_output_shapes
:y
/sequential_4/simple_rnn_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/simple_rnn_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_4/simple_rnn_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)sequential_4/simple_rnn_4/strided_slice_1StridedSlice*sequential_4/simple_rnn_4/Shape_1:output:08sequential_4/simple_rnn_4/strided_slice_1/stack:output:0:sequential_4/simple_rnn_4/strided_slice_1/stack_1:output:0:sequential_4/simple_rnn_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
5sequential_4/simple_rnn_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
'sequential_4/simple_rnn_4/TensorArrayV2TensorListReserve>sequential_4/simple_rnn_4/TensorArrayV2/element_shape:output:02sequential_4/simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Osequential_4/simple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Asequential_4/simple_rnn_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_4/simple_rnn_4/transpose:y:0Xsequential_4/simple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���y
/sequential_4/simple_rnn_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/simple_rnn_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_4/simple_rnn_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)sequential_4/simple_rnn_4/strided_slice_2StridedSlice'sequential_4/simple_rnn_4/transpose:y:08sequential_4/simple_rnn_4/strided_slice_2/stack:output:0:sequential_4/simple_rnn_4/strided_slice_2/stack_1:output:0:sequential_4/simple_rnn_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
Asequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpJsequential_4_simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
2sequential_4/simple_rnn_4/simple_rnn_cell_4/MatMulMatMul2sequential_4/simple_rnn_4/strided_slice_2:output:0Isequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
Bsequential_4/simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpKsequential_4_simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
3sequential_4/simple_rnn_4/simple_rnn_cell_4/BiasAddBiasAdd<sequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul:product:0Jsequential_4/simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
Csequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpLsequential_4_simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0�
4sequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul_1MatMul(sequential_4/simple_rnn_4/zeros:output:0Ksequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
/sequential_4/simple_rnn_4/simple_rnn_cell_4/addAddV2<sequential_4/simple_rnn_4/simple_rnn_cell_4/BiasAdd:output:0>sequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������d�
0sequential_4/simple_rnn_4/simple_rnn_cell_4/ReluRelu3sequential_4/simple_rnn_4/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������d�
7sequential_4/simple_rnn_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
)sequential_4/simple_rnn_4/TensorArrayV2_1TensorListReserve@sequential_4/simple_rnn_4/TensorArrayV2_1/element_shape:output:02sequential_4/simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���`
sequential_4/simple_rnn_4/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2sequential_4/simple_rnn_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������n
,sequential_4/simple_rnn_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_4/simple_rnn_4/whileWhile5sequential_4/simple_rnn_4/while/loop_counter:output:0;sequential_4/simple_rnn_4/while/maximum_iterations:output:0'sequential_4/simple_rnn_4/time:output:02sequential_4/simple_rnn_4/TensorArrayV2_1:handle:0(sequential_4/simple_rnn_4/zeros:output:02sequential_4/simple_rnn_4/strided_slice_1:output:0Qsequential_4/simple_rnn_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0Jsequential_4_simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resourceKsequential_4_simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resourceLsequential_4_simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *7
body/R-
+sequential_4_simple_rnn_4_while_body_202488*7
cond/R-
+sequential_4_simple_rnn_4_while_cond_202487*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
Jsequential_4/simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
<sequential_4/simple_rnn_4/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_4/simple_rnn_4/while:output:3Ssequential_4/simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0�
/sequential_4/simple_rnn_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1sequential_4/simple_rnn_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/simple_rnn_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)sequential_4/simple_rnn_4/strided_slice_3StridedSliceEsequential_4/simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:08sequential_4/simple_rnn_4/strided_slice_3/stack:output:0:sequential_4/simple_rnn_4/strided_slice_3/stack_1:output:0:sequential_4/simple_rnn_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_mask
*sequential_4/simple_rnn_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
%sequential_4/simple_rnn_4/transpose_1	TransposeEsequential_4/simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:03sequential_4/simple_rnn_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d�
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
sequential_4/dense_4/MatMulMatMul2sequential_4/simple_rnn_4/strided_slice_3:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
IdentityIdentity%sequential_4/dense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^sequential_4/dense_4/BiasAdd/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOpC^sequential_4/simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOpB^sequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOpD^sequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp ^sequential_4/simple_rnn_4/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2�
Bsequential_4/simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOpBsequential_4/simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp2�
Asequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOpAsequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp2�
Csequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOpCsequential_4/simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp2B
sequential_4/simple_rnn_4/whilesequential_4/simple_rnn_4/while:_ [
+
_output_shapes
:���������
,
_user_specified_namesimple_rnn_4_input
�
�
H__inference_sequential_4_layer_call_and_return_conditional_losses_203237
simple_rnn_4_input%
simple_rnn_4_203224:d!
simple_rnn_4_203226:d%
simple_rnn_4_203228:dd 
dense_4_203231:d
dense_4_203233:
identity��dense_4/StatefulPartitionedCall�$simple_rnn_4/StatefulPartitionedCall�
$simple_rnn_4/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_4_inputsimple_rnn_4_203224simple_rnn_4_203226simple_rnn_4_203228*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203135�
dense_4/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_4/StatefulPartitionedCall:output:0dense_4_203231dense_4_203233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_202984w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_4/StatefulPartitionedCall%^simple_rnn_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2L
$simple_rnn_4/StatefulPartitionedCall$simple_rnn_4/StatefulPartitionedCall:_ [
+
_output_shapes
:���������
,
_user_specified_namesimple_rnn_4_input
�
�
-__inference_simple_rnn_4_layer_call_fn_203551

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_202966o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_202608

inputs

states0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������dG
ReluReluadd:z:0*
T0*'
_output_shapes
:���������da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������dc

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������d
 
_user_specified_namestates
�
�
while_cond_203927
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_203927___redundant_placeholder04
0while_while_cond_203927___redundant_placeholder14
0while_while_cond_203927___redundant_placeholder24
0while_while_cond_203927___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�	
�
C__inference_dense_4_layer_call_and_return_conditional_losses_202984

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�Q
�
H__inference_sequential_4_layer_call_and_return_conditional_losses_203501

inputsO
=simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resource:dL
>simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resource:dQ
?simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource:dd8
&dense_4_matmul_readvariableop_resource:d5
'dense_4_biasadd_readvariableop_resource:
identity��dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp�4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp�6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp�simple_rnn_4/whileH
simple_rnn_4/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_4/strided_sliceStridedSlicesimple_rnn_4/Shape:output:0)simple_rnn_4/strided_slice/stack:output:0+simple_rnn_4/strided_slice/stack_1:output:0+simple_rnn_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
simple_rnn_4/zeros/packedPack#simple_rnn_4/strided_slice:output:0$simple_rnn_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
simple_rnn_4/zerosFill"simple_rnn_4/zeros/packed:output:0!simple_rnn_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������dp
simple_rnn_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_4/transpose	Transposeinputs$simple_rnn_4/transpose/perm:output:0*
T0*+
_output_shapes
:���������^
simple_rnn_4/Shape_1Shapesimple_rnn_4/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_4/strided_slice_1StridedSlicesimple_rnn_4/Shape_1:output:0+simple_rnn_4/strided_slice_1/stack:output:0-simple_rnn_4/strided_slice_1/stack_1:output:0-simple_rnn_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
simple_rnn_4/TensorArrayV2TensorListReserve1simple_rnn_4/TensorArrayV2/element_shape:output:0%simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Bsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
4simple_rnn_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_4/transpose:y:0Ksimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
"simple_rnn_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_4/strided_slice_2StridedSlicesimple_rnn_4/transpose:y:0+simple_rnn_4/strided_slice_2/stack:output:0-simple_rnn_4/strided_slice_2/stack_1:output:0-simple_rnn_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp=simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
%simple_rnn_4/simple_rnn_cell_4/MatMulMatMul%simple_rnn_4/strided_slice_2:output:0<simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
&simple_rnn_4/simple_rnn_cell_4/BiasAddBiasAdd/simple_rnn_4/simple_rnn_cell_4/MatMul:product:0=simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0�
'simple_rnn_4/simple_rnn_cell_4/MatMul_1MatMulsimple_rnn_4/zeros:output:0>simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
"simple_rnn_4/simple_rnn_cell_4/addAddV2/simple_rnn_4/simple_rnn_cell_4/BiasAdd:output:01simple_rnn_4/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������d�
#simple_rnn_4/simple_rnn_cell_4/ReluRelu&simple_rnn_4/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������d{
*simple_rnn_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
simple_rnn_4/TensorArrayV2_1TensorListReserve3simple_rnn_4/TensorArrayV2_1/element_shape:output:0%simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���S
simple_rnn_4/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������a
simple_rnn_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
simple_rnn_4/whileWhile(simple_rnn_4/while/loop_counter:output:0.simple_rnn_4/while/maximum_iterations:output:0simple_rnn_4/time:output:0%simple_rnn_4/TensorArrayV2_1:handle:0simple_rnn_4/zeros:output:0%simple_rnn_4/strided_slice_1:output:0Dsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resource>simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resource?simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
simple_rnn_4_while_body_203429**
cond"R 
simple_rnn_4_while_cond_203428*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
=simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
/simple_rnn_4/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_4/while:output:3Fsimple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0u
"simple_rnn_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������n
$simple_rnn_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_4/strided_slice_3StridedSlice8simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_4/strided_slice_3/stack:output:0-simple_rnn_4/strided_slice_3/stack_1:output:0-simple_rnn_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maskr
simple_rnn_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_4/transpose_1	Transpose8simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
dense_4/MatMulMatMul%simple_rnn_4/strided_slice_3:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp6^simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp5^simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp7^simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp^simple_rnn_4/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2n
5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp2l
4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp2p
6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp2(
simple_rnn_4/whilesimple_rnn_4/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�4
�
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_202684

inputs*
simple_rnn_cell_4_202609:d&
simple_rnn_cell_4_202611:d*
simple_rnn_cell_4_202613:dd
identity��)simple_rnn_cell_4/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
)simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_4_202609simple_rnn_cell_4_202611simple_rnn_cell_4_202613*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_202608n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_4_202609simple_rnn_cell_4_202611simple_rnn_cell_4_202613*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_202621*
condR
while_cond_202620*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������dz
NoOpNoOp*^simple_rnn_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2V
)simple_rnn_cell_4/StatefulPartitionedCall)simple_rnn_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�Q
�
H__inference_sequential_4_layer_call_and_return_conditional_losses_203387

inputsO
=simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resource:dL
>simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resource:dQ
?simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource:dd8
&dense_4_matmul_readvariableop_resource:d5
'dense_4_biasadd_readvariableop_resource:
identity��dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp�4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp�6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp�simple_rnn_4/whileH
simple_rnn_4/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_4/strided_sliceStridedSlicesimple_rnn_4/Shape:output:0)simple_rnn_4/strided_slice/stack:output:0+simple_rnn_4/strided_slice/stack_1:output:0+simple_rnn_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d�
simple_rnn_4/zeros/packedPack#simple_rnn_4/strided_slice:output:0$simple_rnn_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
simple_rnn_4/zerosFill"simple_rnn_4/zeros/packed:output:0!simple_rnn_4/zeros/Const:output:0*
T0*'
_output_shapes
:���������dp
simple_rnn_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_4/transpose	Transposeinputs$simple_rnn_4/transpose/perm:output:0*
T0*+
_output_shapes
:���������^
simple_rnn_4/Shape_1Shapesimple_rnn_4/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_4/strided_slice_1StridedSlicesimple_rnn_4/Shape_1:output:0+simple_rnn_4/strided_slice_1/stack:output:0-simple_rnn_4/strided_slice_1/stack_1:output:0-simple_rnn_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
simple_rnn_4/TensorArrayV2TensorListReserve1simple_rnn_4/TensorArrayV2/element_shape:output:0%simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Bsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
4simple_rnn_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_4/transpose:y:0Ksimple_rnn_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
"simple_rnn_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_4/strided_slice_2StridedSlicesimple_rnn_4/transpose:y:0+simple_rnn_4/strided_slice_2/stack:output:0-simple_rnn_4/strided_slice_2/stack_1:output:0-simple_rnn_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp=simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
%simple_rnn_4/simple_rnn_cell_4/MatMulMatMul%simple_rnn_4/strided_slice_2:output:0<simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
&simple_rnn_4/simple_rnn_cell_4/BiasAddBiasAdd/simple_rnn_4/simple_rnn_cell_4/MatMul:product:0=simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0�
'simple_rnn_4/simple_rnn_cell_4/MatMul_1MatMulsimple_rnn_4/zeros:output:0>simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
"simple_rnn_4/simple_rnn_cell_4/addAddV2/simple_rnn_4/simple_rnn_cell_4/BiasAdd:output:01simple_rnn_4/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������d�
#simple_rnn_4/simple_rnn_cell_4/ReluRelu&simple_rnn_4/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������d{
*simple_rnn_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
simple_rnn_4/TensorArrayV2_1TensorListReserve3simple_rnn_4/TensorArrayV2_1/element_shape:output:0%simple_rnn_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���S
simple_rnn_4/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������a
simple_rnn_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
simple_rnn_4/whileWhile(simple_rnn_4/while/loop_counter:output:0.simple_rnn_4/while/maximum_iterations:output:0simple_rnn_4/time:output:0%simple_rnn_4/TensorArrayV2_1:handle:0simple_rnn_4/zeros:output:0%simple_rnn_4/strided_slice_1:output:0Dsimple_rnn_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_4_simple_rnn_cell_4_matmul_readvariableop_resource>simple_rnn_4_simple_rnn_cell_4_biasadd_readvariableop_resource?simple_rnn_4_simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
simple_rnn_4_while_body_203315**
cond"R 
simple_rnn_4_while_cond_203314*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
=simple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
/simple_rnn_4/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_4/while:output:3Fsimple_rnn_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0u
"simple_rnn_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������n
$simple_rnn_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_4/strided_slice_3StridedSlice8simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_4/strided_slice_3/stack:output:0-simple_rnn_4/strided_slice_3/stack_1:output:0-simple_rnn_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maskr
simple_rnn_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_4/transpose_1	Transpose8simple_rnn_4/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
dense_4/MatMulMatMul%simple_rnn_4/strided_slice_3:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp6^simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp5^simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp7^simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp^simple_rnn_4/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2n
5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp5simple_rnn_4/simple_rnn_cell_4/BiasAdd/ReadVariableOp2l
4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp4simple_rnn_4/simple_rnn_cell_4/MatMul/ReadVariableOp2p
6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp6simple_rnn_4/simple_rnn_cell_4/MatMul_1/ReadVariableOp2(
simple_rnn_4/whilesimple_rnn_4/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_204075

inputs
states_00
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������dG
ReluReluadd:z:0*
T0*'
_output_shapes
:���������da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������dc

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0
�D
�
+sequential_4_simple_rnn_4_while_body_202488P
Lsequential_4_simple_rnn_4_while_sequential_4_simple_rnn_4_while_loop_counterV
Rsequential_4_simple_rnn_4_while_sequential_4_simple_rnn_4_while_maximum_iterations/
+sequential_4_simple_rnn_4_while_placeholder1
-sequential_4_simple_rnn_4_while_placeholder_11
-sequential_4_simple_rnn_4_while_placeholder_2O
Ksequential_4_simple_rnn_4_while_sequential_4_simple_rnn_4_strided_slice_1_0�
�sequential_4_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0d
Rsequential_4_simple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0:da
Ssequential_4_simple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:df
Tsequential_4_simple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:dd,
(sequential_4_simple_rnn_4_while_identity.
*sequential_4_simple_rnn_4_while_identity_1.
*sequential_4_simple_rnn_4_while_identity_2.
*sequential_4_simple_rnn_4_while_identity_3.
*sequential_4_simple_rnn_4_while_identity_4M
Isequential_4_simple_rnn_4_while_sequential_4_simple_rnn_4_strided_slice_1�
�sequential_4_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_4_tensorarrayunstack_tensorlistfromtensorb
Psequential_4_simple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource:d_
Qsequential_4_simple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource:dd
Rsequential_4_simple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource:dd��Hsequential_4/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�Gsequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp�Isequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
Qsequential_4/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Csequential_4/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_4_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0+sequential_4_simple_rnn_4_while_placeholderZsequential_4/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
Gsequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpRsequential_4_simple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
8sequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMulMatMulJsequential_4/simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem:item:0Osequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
Hsequential_4/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpSsequential_4_simple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0�
9sequential_4/simple_rnn_4/while/simple_rnn_cell_4/BiasAddBiasAddBsequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul:product:0Psequential_4/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
Isequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpTsequential_4_simple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0�
:sequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1MatMul-sequential_4_simple_rnn_4_while_placeholder_2Qsequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
5sequential_4/simple_rnn_4/while/simple_rnn_cell_4/addAddV2Bsequential_4/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd:output:0Dsequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������d�
6sequential_4/simple_rnn_4/while/simple_rnn_cell_4/ReluRelu9sequential_4/simple_rnn_4/while/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������d�
Dsequential_4/simple_rnn_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_4_simple_rnn_4_while_placeholder_1+sequential_4_simple_rnn_4_while_placeholderDsequential_4/simple_rnn_4/while/simple_rnn_cell_4/Relu:activations:0*
_output_shapes
: *
element_dtype0:���g
%sequential_4/simple_rnn_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
#sequential_4/simple_rnn_4/while/addAddV2+sequential_4_simple_rnn_4_while_placeholder.sequential_4/simple_rnn_4/while/add/y:output:0*
T0*
_output_shapes
: i
'sequential_4/simple_rnn_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
%sequential_4/simple_rnn_4/while/add_1AddV2Lsequential_4_simple_rnn_4_while_sequential_4_simple_rnn_4_while_loop_counter0sequential_4/simple_rnn_4/while/add_1/y:output:0*
T0*
_output_shapes
: �
(sequential_4/simple_rnn_4/while/IdentityIdentity)sequential_4/simple_rnn_4/while/add_1:z:0%^sequential_4/simple_rnn_4/while/NoOp*
T0*
_output_shapes
: �
*sequential_4/simple_rnn_4/while/Identity_1IdentityRsequential_4_simple_rnn_4_while_sequential_4_simple_rnn_4_while_maximum_iterations%^sequential_4/simple_rnn_4/while/NoOp*
T0*
_output_shapes
: �
*sequential_4/simple_rnn_4/while/Identity_2Identity'sequential_4/simple_rnn_4/while/add:z:0%^sequential_4/simple_rnn_4/while/NoOp*
T0*
_output_shapes
: �
*sequential_4/simple_rnn_4/while/Identity_3IdentityTsequential_4/simple_rnn_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^sequential_4/simple_rnn_4/while/NoOp*
T0*
_output_shapes
: :����
*sequential_4/simple_rnn_4/while/Identity_4IdentityDsequential_4/simple_rnn_4/while/simple_rnn_cell_4/Relu:activations:0%^sequential_4/simple_rnn_4/while/NoOp*
T0*'
_output_shapes
:���������d�
$sequential_4/simple_rnn_4/while/NoOpNoOpI^sequential_4/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpH^sequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOpJ^sequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_4_simple_rnn_4_while_identity1sequential_4/simple_rnn_4/while/Identity:output:0"a
*sequential_4_simple_rnn_4_while_identity_13sequential_4/simple_rnn_4/while/Identity_1:output:0"a
*sequential_4_simple_rnn_4_while_identity_23sequential_4/simple_rnn_4/while/Identity_2:output:0"a
*sequential_4_simple_rnn_4_while_identity_33sequential_4/simple_rnn_4/while/Identity_3:output:0"a
*sequential_4_simple_rnn_4_while_identity_43sequential_4/simple_rnn_4/while/Identity_4:output:0"�
Isequential_4_simple_rnn_4_while_sequential_4_simple_rnn_4_strided_slice_1Ksequential_4_simple_rnn_4_while_sequential_4_simple_rnn_4_strided_slice_1_0"�
Qsequential_4_simple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resourceSsequential_4_simple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"�
Rsequential_4_simple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resourceTsequential_4_simple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"�
Psequential_4_simple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resourceRsequential_4_simple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0"�
�sequential_4_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor�sequential_4_simple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2�
Hsequential_4/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpHsequential_4/simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2�
Gsequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOpGsequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp2�
Isequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpIsequential_4/simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_202620
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_202620___redundant_placeholder04
0while_while_cond_202620___redundant_placeholder14
0while_while_cond_202620___redundant_placeholder24
0while_while_cond_202620___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_202728

inputs

states0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������dG
ReluReluadd:z:0*
T0*'
_output_shapes
:���������da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������dc

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������d
 
_user_specified_namestates
�
�
H__inference_sequential_4_layer_call_and_return_conditional_losses_202991

inputs%
simple_rnn_4_202967:d!
simple_rnn_4_202969:d%
simple_rnn_4_202971:dd 
dense_4_202985:d
dense_4_202987:
identity��dense_4/StatefulPartitionedCall�$simple_rnn_4/StatefulPartitionedCall�
$simple_rnn_4/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_4_202967simple_rnn_4_202969simple_rnn_4_202971*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_202966�
dense_4/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_4/StatefulPartitionedCall:output:0dense_4_202985dense_4_202987*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_202984w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_4/StatefulPartitionedCall%^simple_rnn_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2L
$simple_rnn_4/StatefulPartitionedCall$simple_rnn_4/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_sequential_4_layer_call_fn_203258

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_202991o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�\
�
"__inference__traced_restore_204240
file_prefix1
assignvariableop_dense_4_kernel:d-
assignvariableop_1_dense_4_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: J
8assignvariableop_7_simple_rnn_4_simple_rnn_cell_4_kernel:dT
Bassignvariableop_8_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel:ddD
6assignvariableop_9_simple_rnn_4_simple_rnn_cell_4_bias:d#
assignvariableop_10_total: #
assignvariableop_11_count: ;
)assignvariableop_12_adam_dense_4_kernel_m:d5
'assignvariableop_13_adam_dense_4_bias_m:R
@assignvariableop_14_adam_simple_rnn_4_simple_rnn_cell_4_kernel_m:d\
Jassignvariableop_15_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_m:ddL
>assignvariableop_16_adam_simple_rnn_4_simple_rnn_cell_4_bias_m:d;
)assignvariableop_17_adam_dense_4_kernel_v:d5
'assignvariableop_18_adam_dense_4_bias_v:R
@assignvariableop_19_adam_simple_rnn_4_simple_rnn_cell_4_kernel_v:d\
Jassignvariableop_20_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_v:ddL
>assignvariableop_21_adam_simple_rnn_4_simple_rnn_cell_4_bias_v:d
identity_23��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp8assignvariableop_7_simple_rnn_4_simple_rnn_cell_4_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpBassignvariableop_8_simple_rnn_4_simple_rnn_cell_4_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp6assignvariableop_9_simple_rnn_4_simple_rnn_cell_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adam_dense_4_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp'assignvariableop_13_adam_dense_4_bias_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp@assignvariableop_14_adam_simple_rnn_4_simple_rnn_cell_4_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpJassignvariableop_15_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp>assignvariableop_16_adam_simple_rnn_4_simple_rnn_cell_4_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_4_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_4_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_simple_rnn_4_simple_rnn_cell_4_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpJassignvariableop_20_adam_simple_rnn_4_simple_rnn_cell_4_recurrent_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp>assignvariableop_21_adam_simple_rnn_4_simple_rnn_cell_4_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_21AssignVariableOp_212(
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
�,
�
while_body_203604
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0:dG
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:dL
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_4_matmul_readvariableop_resource:dE
7while_simple_rnn_cell_4_biasadd_readvariableop_resource:dJ
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:dd��.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_4/MatMul/ReadVariableOp�/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0�
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0�
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������dw
while/simple_rnn_cell_4/ReluReluwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_4/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity*while/simple_rnn_cell_4/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
+sequential_4_simple_rnn_4_while_cond_202487P
Lsequential_4_simple_rnn_4_while_sequential_4_simple_rnn_4_while_loop_counterV
Rsequential_4_simple_rnn_4_while_sequential_4_simple_rnn_4_while_maximum_iterations/
+sequential_4_simple_rnn_4_while_placeholder1
-sequential_4_simple_rnn_4_while_placeholder_11
-sequential_4_simple_rnn_4_while_placeholder_2R
Nsequential_4_simple_rnn_4_while_less_sequential_4_simple_rnn_4_strided_slice_1h
dsequential_4_simple_rnn_4_while_sequential_4_simple_rnn_4_while_cond_202487___redundant_placeholder0h
dsequential_4_simple_rnn_4_while_sequential_4_simple_rnn_4_while_cond_202487___redundant_placeholder1h
dsequential_4_simple_rnn_4_while_sequential_4_simple_rnn_4_while_cond_202487___redundant_placeholder2h
dsequential_4_simple_rnn_4_while_sequential_4_simple_rnn_4_while_cond_202487___redundant_placeholder3,
(sequential_4_simple_rnn_4_while_identity
�
$sequential_4/simple_rnn_4/while/LessLess+sequential_4_simple_rnn_4_while_placeholderNsequential_4_simple_rnn_4_while_less_sequential_4_simple_rnn_4_strided_slice_1*
T0*
_output_shapes
: 
(sequential_4/simple_rnn_4/while/IdentityIdentity(sequential_4/simple_rnn_4/while/Less:z:0*
T0
*
_output_shapes
: "]
(sequential_4_simple_rnn_4_while_identity1sequential_4/simple_rnn_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�,
�
while_body_203712
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0:dG
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:dL
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_4_matmul_readvariableop_resource:dE
7while_simple_rnn_cell_4_biasadd_readvariableop_resource:dJ
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:dd��.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_4/MatMul/ReadVariableOp�/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0�
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0�
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������dw
while/simple_rnn_cell_4/ReluReluwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_4/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity*while/simple_rnn_cell_4/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�,
�
while_body_203928
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_4_matmul_readvariableop_resource_0:dG
9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:dL
:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_4_matmul_readvariableop_resource:dE
7while_simple_rnn_cell_4_biasadd_readvariableop_resource:dJ
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:dd��.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_4/MatMul/ReadVariableOp�/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
-while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
while/simple_rnn_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0�
while/simple_rnn_cell_4/BiasAddBiasAdd(while/simple_rnn_cell_4/MatMul:product:06while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0�
 while/simple_rnn_cell_4/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
while/simple_rnn_cell_4/addAddV2(while/simple_rnn_cell_4/BiasAdd:output:0*while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������dw
while/simple_rnn_cell_4/ReluReluwhile/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������d�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_4/Relu:activations:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity*while/simple_rnn_cell_4/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:���������d�

while/NoOpNoOp/^while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_4/MatMul/ReadVariableOp0^while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_4_biasadd_readvariableop_resource9while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_4_matmul_1_readvariableop_resource:while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_4_matmul_readvariableop_resource8while_simple_rnn_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2`
.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp.while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_4/MatMul/ReadVariableOp-while/simple_rnn_cell_4/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_sequential_4_layer_call_and_return_conditional_losses_203221
simple_rnn_4_input%
simple_rnn_4_203208:d!
simple_rnn_4_203210:d%
simple_rnn_4_203212:dd 
dense_4_203215:d
dense_4_203217:
identity��dense_4/StatefulPartitionedCall�$simple_rnn_4/StatefulPartitionedCall�
$simple_rnn_4/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_4_inputsimple_rnn_4_203208simple_rnn_4_203210simple_rnn_4_203212*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_202966�
dense_4/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_4/StatefulPartitionedCall:output:0dense_4_203215dense_4_203217*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_202984w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_4/StatefulPartitionedCall%^simple_rnn_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2L
$simple_rnn_4/StatefulPartitionedCall$simple_rnn_4/StatefulPartitionedCall:_ [
+
_output_shapes
:���������
,
_user_specified_namesimple_rnn_4_input
�8
�
simple_rnn_4_while_body_2034296
2simple_rnn_4_while_simple_rnn_4_while_loop_counter<
8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations"
simple_rnn_4_while_placeholder$
 simple_rnn_4_while_placeholder_1$
 simple_rnn_4_while_placeholder_25
1simple_rnn_4_while_simple_rnn_4_strided_slice_1_0q
msimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0:dT
Fsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0:dY
Gsimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0:dd
simple_rnn_4_while_identity!
simple_rnn_4_while_identity_1!
simple_rnn_4_while_identity_2!
simple_rnn_4_while_identity_3!
simple_rnn_4_while_identity_43
/simple_rnn_4_while_simple_rnn_4_strided_slice_1o
ksimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource:dR
Dsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource:dW
Esimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource:dd��;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp�:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp�<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp�
Dsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
6simple_rnn_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_4_while_placeholderMsimple_rnn_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0*
_output_shapes

:d*
dtype0�
+simple_rnn_4/while/simple_rnn_cell_4/MatMulMatMul=simple_rnn_4/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0�
,simple_rnn_4/while/simple_rnn_cell_4/BiasAddBiasAdd5simple_rnn_4/while/simple_rnn_cell_4/MatMul:product:0Csimple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0�
-simple_rnn_4/while/simple_rnn_cell_4/MatMul_1MatMul simple_rnn_4_while_placeholder_2Dsimple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
(simple_rnn_4/while/simple_rnn_cell_4/addAddV25simple_rnn_4/while/simple_rnn_cell_4/BiasAdd:output:07simple_rnn_4/while/simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������d�
)simple_rnn_4/while/simple_rnn_cell_4/ReluRelu,simple_rnn_4/while/simple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������d�
7simple_rnn_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_4_while_placeholder_1simple_rnn_4_while_placeholder7simple_rnn_4/while/simple_rnn_cell_4/Relu:activations:0*
_output_shapes
: *
element_dtype0:���Z
simple_rnn_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_4/while/addAddV2simple_rnn_4_while_placeholder!simple_rnn_4/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_4/while/add_1AddV22simple_rnn_4_while_simple_rnn_4_while_loop_counter#simple_rnn_4/while/add_1/y:output:0*
T0*
_output_shapes
: �
simple_rnn_4/while/IdentityIdentitysimple_rnn_4/while/add_1:z:0^simple_rnn_4/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_4/while/Identity_1Identity8simple_rnn_4_while_simple_rnn_4_while_maximum_iterations^simple_rnn_4/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_4/while/Identity_2Identitysimple_rnn_4/while/add:z:0^simple_rnn_4/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_4/while/Identity_3IdentityGsimple_rnn_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_4/while/NoOp*
T0*
_output_shapes
: :����
simple_rnn_4/while/Identity_4Identity7simple_rnn_4/while/simple_rnn_cell_4/Relu:activations:0^simple_rnn_4/while/NoOp*
T0*'
_output_shapes
:���������d�
simple_rnn_4/while/NoOpNoOp<^simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;^simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp=^simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_4_while_identity$simple_rnn_4/while/Identity:output:0"G
simple_rnn_4_while_identity_1&simple_rnn_4/while/Identity_1:output:0"G
simple_rnn_4_while_identity_2&simple_rnn_4/while/Identity_2:output:0"G
simple_rnn_4_while_identity_3&simple_rnn_4/while/Identity_3:output:0"G
simple_rnn_4_while_identity_4&simple_rnn_4/while/Identity_4:output:0"d
/simple_rnn_4_while_simple_rnn_4_strided_slice_11simple_rnn_4_while_simple_rnn_4_strided_slice_1_0"�
Dsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resourceFsimple_rnn_4_while_simple_rnn_cell_4_biasadd_readvariableop_resource_0"�
Esimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resourceGsimple_rnn_4_while_simple_rnn_cell_4_matmul_1_readvariableop_resource_0"�
Csimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resourceEsimple_rnn_4_while_simple_rnn_cell_4_matmul_readvariableop_resource_0"�
ksimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensormsimple_rnn_4_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2z
;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp;simple_rnn_4/while/simple_rnn_cell_4/BiasAdd/ReadVariableOp2x
:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp:simple_rnn_4/while/simple_rnn_cell_4/MatMul/ReadVariableOp2|
<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp<simple_rnn_4/while/simple_rnn_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_sequential_4_layer_call_fn_203205
simple_rnn_4_input
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_203177o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:���������
,
_user_specified_namesimple_rnn_4_input
�=
�
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203670
inputs_0B
0simple_rnn_cell_4_matmul_readvariableop_resource:d?
1simple_rnn_cell_4_biasadd_readvariableop_resource:dD
2simple_rnn_cell_4_matmul_1_readvariableop_resource:dd
identity��(simple_rnn_cell_4/BiasAdd/ReadVariableOp�'simple_rnn_cell_4/MatMul/ReadVariableOp�)simple_rnn_cell_4/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0�
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������dk
simple_rnn_cell_4/ReluRelusimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_203604*
condR
while_cond_203603*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
while_cond_203711
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_203711___redundant_placeholder04
0while_while_cond_203711___redundant_placeholder14
0while_while_cond_203711___redundant_placeholder24
0while_while_cond_203711___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�
�
H__inference_sequential_4_layer_call_and_return_conditional_losses_203177

inputs%
simple_rnn_4_203164:d!
simple_rnn_4_203166:d%
simple_rnn_4_203168:dd 
dense_4_203171:d
dense_4_203173:
identity��dense_4/StatefulPartitionedCall�$simple_rnn_4/StatefulPartitionedCall�
$simple_rnn_4/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_4_203164simple_rnn_4_203166simple_rnn_4_203168*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203135�
dense_4/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_4/StatefulPartitionedCall:output:0dense_4_203171dense_4_203173*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_202984w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_4/StatefulPartitionedCall%^simple_rnn_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2L
$simple_rnn_4/StatefulPartitionedCall$simple_rnn_4/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_202899
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_202899___redundant_placeholder04
0while_while_cond_202899___redundant_placeholder14
0while_while_cond_202899___redundant_placeholder24
0while_while_cond_202899___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:
�

�
2__inference_simple_rnn_cell_4_layer_call_fn_204027

inputs
states_0
unknown:d
	unknown_0:d
	unknown_1:dd
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_202608o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������d
"
_user_specified_name
states/0
�=
�
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203994

inputsB
0simple_rnn_cell_4_matmul_readvariableop_resource:d?
1simple_rnn_cell_4_biasadd_readvariableop_resource:dD
2simple_rnn_cell_4_matmul_1_readvariableop_resource:dd
identity��(simple_rnn_cell_4/BiasAdd/ReadVariableOp�'simple_rnn_cell_4/MatMul/ReadVariableOp�)simple_rnn_cell_4/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0�
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������dk
simple_rnn_cell_4/ReluRelusimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_203928*
condR
while_cond_203927*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�=
�
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203778
inputs_0B
0simple_rnn_cell_4_matmul_readvariableop_resource:d?
1simple_rnn_cell_4_biasadd_readvariableop_resource:dD
2simple_rnn_cell_4_matmul_1_readvariableop_resource:dd
identity��(simple_rnn_cell_4/BiasAdd/ReadVariableOp�'simple_rnn_cell_4/MatMul/ReadVariableOp�)simple_rnn_cell_4/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0�
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������dk
simple_rnn_cell_4/ReluRelusimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_203712*
condR
while_cond_203711*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�4
�
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_202843

inputs*
simple_rnn_cell_4_202768:d&
simple_rnn_cell_4_202770:d*
simple_rnn_cell_4_202772:dd
identity��)simple_rnn_cell_4/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
)simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_4_202768simple_rnn_cell_4_202770simple_rnn_cell_4_202772*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_202728n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_4_202768simple_rnn_cell_4_202770simple_rnn_cell_4_202772*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_202780*
condR
while_cond_202779*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������dz
NoOpNoOp*^simple_rnn_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2V
)simple_rnn_cell_4/StatefulPartitionedCall)simple_rnn_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�!
�
while_body_202780
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_4_202802_0:d.
 while_simple_rnn_cell_4_202804_0:d2
 while_simple_rnn_cell_4_202806_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_4_202802:d,
while_simple_rnn_cell_4_202804:d0
while_simple_rnn_cell_4_202806:dd��/while/simple_rnn_cell_4/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
/while/simple_rnn_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_4_202802_0 while_simple_rnn_cell_4_202804_0 while_simple_rnn_cell_4_202806_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������d:���������d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_202728�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity8while/simple_rnn_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������d~

while/NoOpNoOp0^while/simple_rnn_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"B
while_simple_rnn_cell_4_202802 while_simple_rnn_cell_4_202802_0"B
while_simple_rnn_cell_4_202804 while_simple_rnn_cell_4_202804_0"B
while_simple_rnn_cell_4_202806 while_simple_rnn_cell_4_202806_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������d: : : : : 2b
/while/simple_rnn_cell_4/StatefulPartitionedCall/while/simple_rnn_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_sequential_4_layer_call_fn_203004
simple_rnn_4_input
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_202991o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:���������
,
_user_specified_namesimple_rnn_4_input
�
�
-__inference_sequential_4_layer_call_fn_203273

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_203177o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�=
�
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203135

inputsB
0simple_rnn_cell_4_matmul_readvariableop_resource:d?
1simple_rnn_cell_4_biasadd_readvariableop_resource:dD
2simple_rnn_cell_4_matmul_1_readvariableop_resource:dd
identity��(simple_rnn_cell_4/BiasAdd/ReadVariableOp�'simple_rnn_cell_4/MatMul/ReadVariableOp�)simple_rnn_cell_4/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
'simple_rnn_cell_4/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
simple_rnn_cell_4/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
(simple_rnn_cell_4/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
simple_rnn_cell_4/BiasAddBiasAdd"simple_rnn_cell_4/MatMul:product:00simple_rnn_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
)simple_rnn_cell_4/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_4_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0�
simple_rnn_cell_4/MatMul_1MatMulzeros:output:01simple_rnn_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
simple_rnn_cell_4/addAddV2"simple_rnn_cell_4/BiasAdd:output:0$simple_rnn_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:���������dk
simple_rnn_cell_4/ReluRelusimple_rnn_cell_4/add:z:0*
T0*'
_output_shapes
:���������dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_4_matmul_readvariableop_resource1simple_rnn_cell_4_biasadd_readvariableop_resource2simple_rnn_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_203069*
condR
while_cond_203068*8
output_shapes'
%: : : : :���������d: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����d   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������d�
NoOpNoOp)^simple_rnn_cell_4/BiasAdd/ReadVariableOp(^simple_rnn_cell_4/MatMul/ReadVariableOp*^simple_rnn_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2T
(simple_rnn_cell_4/BiasAdd/ReadVariableOp(simple_rnn_cell_4/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_4/MatMul/ReadVariableOp'simple_rnn_cell_4/MatMul/ReadVariableOp2V
)simple_rnn_cell_4/MatMul_1/ReadVariableOp)simple_rnn_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_203603
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_203603___redundant_placeholder04
0while_while_cond_203603___redundant_placeholder14
0while_while_cond_203603___redundant_placeholder24
0while_while_cond_203603___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������d:

_output_shapes
: :

_output_shapes
:"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
U
simple_rnn_4_input?
$serving_default_simple_rnn_4_input:0���������;
dense_40
StatefulPartitionedCall:0���������tensorflow/serving/predict:�c
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures"
_tf_keras_sequential
�
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
iter

beta_1

beta_2
	decay
 learning_ratemFmG!mH"mI#mJvKvL!vM"vN#vO"
	optimizer
C
!0
"1
#2
3
4"
trackable_list_wrapper
C
!0
"1
#2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_sequential_4_layer_call_fn_203004
-__inference_sequential_4_layer_call_fn_203258
-__inference_sequential_4_layer_call_fn_203273
-__inference_sequential_4_layer_call_fn_203205�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_sequential_4_layer_call_and_return_conditional_losses_203387
H__inference_sequential_4_layer_call_and_return_conditional_losses_203501
H__inference_sequential_4_layer_call_and_return_conditional_losses_203221
H__inference_sequential_4_layer_call_and_return_conditional_losses_203237�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
!__inference__wrapped_model_202560simple_rnn_4_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
)serving_default"
signature_map
�

!kernel
"recurrent_kernel
#bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
._random_generator
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

1states
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_simple_rnn_4_layer_call_fn_203529
-__inference_simple_rnn_4_layer_call_fn_203540
-__inference_simple_rnn_4_layer_call_fn_203551
-__inference_simple_rnn_4_layer_call_fn_203562�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203670
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203778
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203886
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203994�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 :d2dense_4/kernel
:2dense_4/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_dense_4_layer_call_fn_204003�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_4_layer_call_and_return_conditional_losses_204013�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
7:5d2%simple_rnn_4/simple_rnn_cell_4/kernel
A:?dd2/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel
1:/d2#simple_rnn_4/simple_rnn_cell_4/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_signature_wrapper_203518simple_rnn_4_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
!0
"1
#2"
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
*	variables
+trainable_variables
,regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
2__inference_simple_rnn_cell_4_layer_call_fn_204027
2__inference_simple_rnn_cell_4_layer_call_fn_204041�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_204058
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_204075�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
N
	Btotal
	Ccount
D	variables
E	keras_api"
_tf_keras_metric
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
.
B0
C1"
trackable_list_wrapper
-
D	variables"
_generic_user_object
%:#d2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
<::d2,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/m
F:Ddd26Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/m
6:4d2*Adam/simple_rnn_4/simple_rnn_cell_4/bias/m
%:#d2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
<::d2,Adam/simple_rnn_4/simple_rnn_cell_4/kernel/v
F:Ddd26Adam/simple_rnn_4/simple_rnn_cell_4/recurrent_kernel/v
6:4d2*Adam/simple_rnn_4/simple_rnn_cell_4/bias/v�
!__inference__wrapped_model_202560{!#"?�<
5�2
0�-
simple_rnn_4_input���������
� "1�.
,
dense_4!�
dense_4����������
C__inference_dense_4_layer_call_and_return_conditional_losses_204013\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� {
(__inference_dense_4_layer_call_fn_204003O/�,
%�"
 �
inputs���������d
� "�����������
H__inference_sequential_4_layer_call_and_return_conditional_losses_203221w!#"G�D
=�:
0�-
simple_rnn_4_input���������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_4_layer_call_and_return_conditional_losses_203237w!#"G�D
=�:
0�-
simple_rnn_4_input���������
p

 
� "%�"
�
0���������
� �
H__inference_sequential_4_layer_call_and_return_conditional_losses_203387k!#";�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_4_layer_call_and_return_conditional_losses_203501k!#";�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
-__inference_sequential_4_layer_call_fn_203004j!#"G�D
=�:
0�-
simple_rnn_4_input���������
p 

 
� "�����������
-__inference_sequential_4_layer_call_fn_203205j!#"G�D
=�:
0�-
simple_rnn_4_input���������
p

 
� "�����������
-__inference_sequential_4_layer_call_fn_203258^!#";�8
1�.
$�!
inputs���������
p 

 
� "�����������
-__inference_sequential_4_layer_call_fn_203273^!#";�8
1�.
$�!
inputs���������
p

 
� "�����������
$__inference_signature_wrapper_203518�!#"U�R
� 
K�H
F
simple_rnn_4_input0�-
simple_rnn_4_input���������"1�.
,
dense_4!�
dense_4����������
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203670}!#"O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"
�
0���������d
� �
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203778}!#"O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"
�
0���������d
� �
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203886m!#"?�<
5�2
$�!
inputs���������

 
p 

 
� "%�"
�
0���������d
� �
H__inference_simple_rnn_4_layer_call_and_return_conditional_losses_203994m!#"?�<
5�2
$�!
inputs���������

 
p

 
� "%�"
�
0���������d
� �
-__inference_simple_rnn_4_layer_call_fn_203529p!#"O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "����������d�
-__inference_simple_rnn_4_layer_call_fn_203540p!#"O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "����������d�
-__inference_simple_rnn_4_layer_call_fn_203551`!#"?�<
5�2
$�!
inputs���������

 
p 

 
� "����������d�
-__inference_simple_rnn_4_layer_call_fn_203562`!#"?�<
5�2
$�!
inputs���������

 
p

 
� "����������d�
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_204058�!#"\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������d
p 
� "R�O
H�E
�
0/0���������d
$�!
�
0/1/0���������d
� �
M__inference_simple_rnn_cell_4_layer_call_and_return_conditional_losses_204075�!#"\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������d
p
� "R�O
H�E
�
0/0���������d
$�!
�
0/1/0���������d
� �
2__inference_simple_rnn_cell_4_layer_call_fn_204027�!#"\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������d
p 
� "D�A
�
0���������d
"�
�
1/0���������d�
2__inference_simple_rnn_cell_4_layer_call_fn_204041�!#"\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������d
p
� "D�A
�
0���������d
"�
�
1/0���������d