ٮ
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
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
executor_typestring �
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.1-0-g85c8b2a817f8��
�
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameembedding_1/embeddings
�
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes
:	�*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
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
�
"RMSprop/embedding_1/embeddings/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"RMSprop/embedding_1/embeddings/rms
�
6RMSprop/embedding_1/embeddings/rms/Read/ReadVariableOpReadVariableOp"RMSprop/embedding_1/embeddings/rms*
_output_shapes
:	�*
dtype0
�
RMSprop/dense_2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameRMSprop/dense_2/kernel/rms
�
.RMSprop/dense_2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/kernel/rms*
_output_shapes

:*
dtype0
�
RMSprop/dense_2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_2/bias/rms
�
,RMSprop/dense_2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/bias/rms*
_output_shapes
:*
dtype0
�
RMSprop/dense_3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameRMSprop/dense_3/kernel/rms
�
.RMSprop/dense_3/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_3/kernel/rms*
_output_shapes

:*
dtype0
�
RMSprop/dense_3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_3/bias/rms
�
,RMSprop/dense_3/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_3/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
�"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�!
value�!B�! B�!
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
b

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
v
%iter
	&decay
'learning_rate
(momentum
)rho	rmsS	rmsT	rmsU	rmsV	 rmsW
 
#
0
1
2
3
 4
#
0
1
2
3
 4
�
regularization_losses
	variables
*layer_regularization_losses
	trainable_variables
+non_trainable_variables

,layers
-metrics
.layer_metrics
 
fd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
�
regularization_losses
	variables
/layer_regularization_losses
trainable_variables
0non_trainable_variables

1layers
2metrics
3layer_metrics
 
 
 
�
regularization_losses
	variables
4layer_regularization_losses
trainable_variables
5non_trainable_variables

6layers
7metrics
8layer_metrics
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
	variables
9layer_regularization_losses
trainable_variables
:non_trainable_variables

;layers
<metrics
=layer_metrics
 
 
 
�
regularization_losses
	variables
>layer_regularization_losses
trainable_variables
?non_trainable_variables

@layers
Ametrics
Blayer_metrics
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
�
!regularization_losses
"	variables
Clayer_regularization_losses
#trainable_variables
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_metrics
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
 
 
#
0
1
2
3
4

H0
I1
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
	Jtotal
	Kcount
L	variables
M	keras_api
D
	Ntotal
	Ocount
P
_fn_kwargs
Q	variables
R	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

L	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

N0
O1

Q	variables
��
VARIABLE_VALUE"RMSprop/embedding_1/embeddings/rmsXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/dense_2/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUERMSprop/dense_2/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/dense_3/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUERMSprop/dense_3/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
h
serving_default_input_2Placeholder*
_output_shapes

:*
dtype0*
shape
:
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2embedding_1/embeddingsdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_23971
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_1/embeddings/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp6RMSprop/embedding_1/embeddings/rms/Read/ReadVariableOp.RMSprop/dense_2/kernel/rms/Read/ReadVariableOp,RMSprop/dense_2/bias/rms/Read/ReadVariableOp.RMSprop/dense_3/kernel/rms/Read/ReadVariableOp,RMSprop/dense_3/bias/rms/Read/ReadVariableOpConst* 
Tin
2	*
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
GPU2*0J 8� *'
f"R 
__inference__traced_save_24334
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_1/embeddingsdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1"RMSprop/embedding_1/embeddings/rmsRMSprop/dense_2/kernel/rmsRMSprop/dense_2/bias/rmsRMSprop/dense_3/kernel/rmsRMSprop/dense_3/bias/rms*
Tin
2*
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
GPU2*0J 8� **
f%R#
!__inference__traced_restore_24401�
�
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_23933

inputs
embedding_1_23917
dense_2_23921
dense_2_23923
dense_3_23927
dense_3_23929
identity��dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_23917*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_237492%
#embedding_1/StatefulPartitionedCall�
*global_average_pooling1d_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_237662,
*global_average_pooling1d_1/PartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_2_23921dense_2_23923*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_237842!
dense_2/StatefulPartitionedCall�
dropout_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_238172
dropout_1/PartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_3_23927dense_3_23929*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_238412!
dense_3/StatefulPartitionedCall�
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
�	
�
B__inference_dense_3_layer_call_and_return_conditional_losses_23841

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddX
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes

:2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
�	
�
B__inference_dense_2_layer_call_and_return_conditional_losses_24198

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddO
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
�
E
)__inference_dropout_1_layer_call_fn_24234

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_238172
PartitionedCallc
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes

::F B

_output_shapes

:
 
_user_specified_nameinputs
�
V
:__inference_global_average_pooling1d_1_layer_call_fn_24176

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_237332
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_23766

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesf
MeanMeaninputsMean/reduction_indices:output:0*
T0*
_output_shapes

:2
MeanX
IdentityIdentityMean:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
,__inference_sequential_1_layer_call_fn_24060
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_239332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes

:
!
_user_specified_name	input_2
�
�
F__inference_embedding_1_layer_call_and_return_conditional_losses_23749

inputs
embedding_lookup_23743
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_23743inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/23743*"
_output_shapes
:*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/23743*"
_output_shapes
:2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*"
_output_shapes
:2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*"
_output_shapes
:2

Identity"
identityIdentity:output:0*!
_input_shapes
::2$
embedding_lookupembedding_lookup:F B

_output_shapes

:
 
_user_specified_nameinputs
�
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_24030
input_2&
"embedding_1_embedding_lookup_24007*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity��dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�embedding_1/embedding_lookup�
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_24007input_2",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/24007*"
_output_shapes
:*
dtype02
embedding_1/embedding_lookup�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/24007*"
_output_shapes
:2'
%embedding_1/embedding_lookup/Identity�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*"
_output_shapes
:2)
'embedding_1/embedding_lookup/Identity_1�
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_1/Mean/reduction_indices�
global_average_pooling1d_1/MeanMean0embedding_1/embedding_lookup/Identity_1:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*
_output_shapes

:2!
global_average_pooling1d_1/Mean�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMul(global_average_pooling1d_1/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAddg
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_2/Reluy
dropout_1/IdentityIdentitydense_2/Relu:activations:0*
T0*
_output_shapes

:2
dropout_1/Identity�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMuldropout_1/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_3/BiasAddp
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_3/Softmax�
IdentityIdentitydense_3/Softmax:softmax:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:G C

_output_shapes

:
!
_user_specified_name	input_2
�
�
,__inference_sequential_1_layer_call_fn_24134

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_238992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
�(
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_24093

inputs&
"embedding_1_embedding_lookup_24063*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity��dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�embedding_1/embedding_lookup�
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_24063inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/24063*"
_output_shapes
:*
dtype02
embedding_1/embedding_lookup�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/24063*"
_output_shapes
:2'
%embedding_1/embedding_lookup/Identity�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*"
_output_shapes
:2)
'embedding_1/embedding_lookup/Identity_1�
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_1/Mean/reduction_indices�
global_average_pooling1d_1/MeanMean0embedding_1/embedding_lookup/Identity_1:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*
_output_shapes

:2!
global_average_pooling1d_1/Mean�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMul(global_average_pooling1d_1/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAddg
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_2/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_1/dropout/Const�
dropout_1/dropout/MulMuldense_2/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*
_output_shapes

:2
dropout_1/dropout/Mul�
dropout_1/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout_1/dropout/Shape�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*
_output_shapes

:*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform�
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2"
 dropout_1/dropout/GreaterEqual/y�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes

:2 
dropout_1/dropout/GreaterEqual�
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
dropout_1/dropout/Cast�
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*
_output_shapes

:2
dropout_1/dropout/Mul_1�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_3/BiasAddp
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_3/Softmax�
IdentityIdentitydense_3/Softmax:softmax:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:F B

_output_shapes

:
 
_user_specified_nameinputs
�

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_23812

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constj
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes

:2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes

:*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes

:2
dropout/GreaterEqualv
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
dropout/Castq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes

:2
dropout/Mul_1\
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes

::F B

_output_shapes

:
 
_user_specified_nameinputs
�
�
,__inference_sequential_1_layer_call_fn_24149

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_239332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
�
q
+__inference_embedding_1_layer_call_fn_24165

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_237492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:2

Identity"
identityIdentity:output:0*!
_input_shapes
::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
�
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_24119

inputs&
"embedding_1_embedding_lookup_24096*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity��dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�embedding_1/embedding_lookup�
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_24096inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/24096*"
_output_shapes
:*
dtype02
embedding_1/embedding_lookup�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/24096*"
_output_shapes
:2'
%embedding_1/embedding_lookup/Identity�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*"
_output_shapes
:2)
'embedding_1/embedding_lookup/Identity_1�
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_1/Mean/reduction_indices�
global_average_pooling1d_1/MeanMean0embedding_1/embedding_lookup/Identity_1:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*
_output_shapes

:2!
global_average_pooling1d_1/Mean�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMul(global_average_pooling1d_1/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAddg
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_2/Reluy
dropout_1/IdentityIdentitydense_2/Relu:activations:0*
T0*
_output_shapes

:2
dropout_1/Identity�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMuldropout_1/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_3/BiasAddp
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_3/Softmax�
IdentityIdentitydense_3/Softmax:softmax:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:F B

_output_shapes

:
 
_user_specified_nameinputs
�'
�
 __inference__wrapped_model_23717
input_23
/sequential_1_embedding_1_embedding_lookup_236947
3sequential_1_dense_2_matmul_readvariableop_resource8
4sequential_1_dense_2_biasadd_readvariableop_resource7
3sequential_1_dense_3_matmul_readvariableop_resource8
4sequential_1_dense_3_biasadd_readvariableop_resource
identity��+sequential_1/dense_2/BiasAdd/ReadVariableOp�*sequential_1/dense_2/MatMul/ReadVariableOp�+sequential_1/dense_3/BiasAdd/ReadVariableOp�*sequential_1/dense_3/MatMul/ReadVariableOp�)sequential_1/embedding_1/embedding_lookup�
)sequential_1/embedding_1/embedding_lookupResourceGather/sequential_1_embedding_1_embedding_lookup_23694input_2",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@sequential_1/embedding_1/embedding_lookup/23694*"
_output_shapes
:*
dtype02+
)sequential_1/embedding_1/embedding_lookup�
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@sequential_1/embedding_1/embedding_lookup/23694*"
_output_shapes
:24
2sequential_1/embedding_1/embedding_lookup/Identity�
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*"
_output_shapes
:26
4sequential_1/embedding_1/embedding_lookup/Identity_1�
>sequential_1/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_1/global_average_pooling1d_1/Mean/reduction_indices�
,sequential_1/global_average_pooling1d_1/MeanMean=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0Gsequential_1/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*
_output_shapes

:2.
,sequential_1/global_average_pooling1d_1/Mean�
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_1/dense_2/MatMul/ReadVariableOp�
sequential_1/dense_2/MatMulMatMul5sequential_1/global_average_pooling1d_1/Mean:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
sequential_1/dense_2/MatMul�
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOp�
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
sequential_1/dense_2/BiasAdd�
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*
_output_shapes

:2
sequential_1/dense_2/Relu�
sequential_1/dropout_1/IdentityIdentity'sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes

:2!
sequential_1/dropout_1/Identity�
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*sequential_1/dense_3/MatMul/ReadVariableOp�
sequential_1/dense_3/MatMulMatMul(sequential_1/dropout_1/Identity:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
sequential_1/dense_3/MatMul�
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOp�
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
sequential_1/dense_3/BiasAdd�
sequential_1/dense_3/SoftmaxSoftmax%sequential_1/dense_3/BiasAdd:output:0*
T0*
_output_shapes

:2
sequential_1/dense_3/Softmax�
IdentityIdentity&sequential_1/dense_3/Softmax:softmax:0,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp*^sequential_1/embedding_1/embedding_lookup*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::::2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp2V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup:G C

_output_shapes

:
!
_user_specified_name	input_2
�(
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_24004
input_2&
"embedding_1_embedding_lookup_23974*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity��dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�embedding_1/embedding_lookup�
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_23974input_2",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/23974*"
_output_shapes
:*
dtype02
embedding_1/embedding_lookup�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/23974*"
_output_shapes
:2'
%embedding_1/embedding_lookup/Identity�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*"
_output_shapes
:2)
'embedding_1/embedding_lookup/Identity_1�
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_1/Mean/reduction_indices�
global_average_pooling1d_1/MeanMean0embedding_1/embedding_lookup/Identity_1:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*
_output_shapes

:2!
global_average_pooling1d_1/Mean�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMul(global_average_pooling1d_1/Mean:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAddg
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_2/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_1/dropout/Const�
dropout_1/dropout/MulMuldense_2/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*
_output_shapes

:2
dropout_1/dropout/Mul�
dropout_1/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout_1/dropout/Shape�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*
_output_shapes

:*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform�
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2"
 dropout_1/dropout/GreaterEqual/y�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes

:2 
dropout_1/dropout/GreaterEqual�
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
dropout_1/dropout/Cast�
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*
_output_shapes

:2
dropout_1/dropout/Mul_1�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_3/BiasAddp
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_3/Softmax�
IdentityIdentitydense_3/Softmax:softmax:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:G C

_output_shapes

:
!
_user_specified_name	input_2
�	
�
B__inference_dense_3_layer_call_and_return_conditional_losses_24245

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddX
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes

:2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
�
b
)__inference_dropout_1_layer_call_fn_24229

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_238122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes

:22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
�
|
'__inference_dense_3_layer_call_fn_24254

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_238412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
�/
�
__inference__traced_save_24334
file_prefix5
1savev2_embedding_1_embeddings_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopA
=savev2_rmsprop_embedding_1_embeddings_rms_read_readvariableop9
5savev2_rmsprop_dense_2_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_2_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_3_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_3_bias_rms_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_1_embeddings_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop=savev2_rmsprop_embedding_1_embeddings_rms_read_readvariableop5savev2_rmsprop_dense_2_kernel_rms_read_readvariableop3savev2_rmsprop_dense_2_bias_rms_read_readvariableop5savev2_rmsprop_dense_3_kernel_rms_read_readvariableop3savev2_rmsprop_dense_3_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*
_input_shapesn
l: :	�::::: : : : : : : : : :	�::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :
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
: :%!

_output_shapes
:	�:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
�
�
F__inference_embedding_1_layer_call_and_return_conditional_losses_24158

inputs
embedding_lookup_24152
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_24152inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/24152*"
_output_shapes
:*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/24152*"
_output_shapes
:2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*"
_output_shapes
:2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*"
_output_shapes
:2

Identity"
identityIdentity:output:0*!
_input_shapes
::2$
embedding_lookupembedding_lookup:F B

_output_shapes

:
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_24171

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_24182

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesf
MeanMeaninputsMean/reduction_indices:output:0*
T0*
_output_shapes

:2
MeanX
IdentityIdentityMean:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_23971
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_237172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes

:
!
_user_specified_name	input_2
�	
�
B__inference_dense_2_layer_call_and_return_conditional_losses_23784

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddO
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_23733

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
G__inference_sequential_1_layer_call_and_return_conditional_losses_23899

inputs
embedding_1_23883
dense_2_23887
dense_2_23889
dense_3_23893
dense_3_23895
identity��dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_23883*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_237492%
#embedding_1/StatefulPartitionedCall�
*global_average_pooling1d_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_237662,
*global_average_pooling1d_1/PartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_2_23887dense_2_23889*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_237842!
dense_2/StatefulPartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_238122#
!dropout_1/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_3_23893dense_3_23895*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_238412!
dense_3/StatefulPartitionedCall�
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
�
|
'__inference_dense_2_layer_call_fn_24207

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_237842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*%
_input_shapes
:::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
�R
�	
!__inference__traced_restore_24401
file_prefix+
'assignvariableop_embedding_1_embeddings%
!assignvariableop_1_dense_2_kernel#
assignvariableop_2_dense_2_bias%
!assignvariableop_3_dense_3_kernel#
assignvariableop_4_dense_3_bias#
assignvariableop_5_rmsprop_iter$
 assignvariableop_6_rmsprop_decay,
(assignvariableop_7_rmsprop_learning_rate'
#assignvariableop_8_rmsprop_momentum"
assignvariableop_9_rmsprop_rho
assignvariableop_10_total
assignvariableop_11_count
assignvariableop_12_total_1
assignvariableop_13_count_1:
6assignvariableop_14_rmsprop_embedding_1_embeddings_rms2
.assignvariableop_15_rmsprop_dense_2_kernel_rms0
,assignvariableop_16_rmsprop_dense_2_bias_rms2
.assignvariableop_17_rmsprop_dense_3_kernel_rms0
,assignvariableop_18_rmsprop_dense_3_bias_rms
identity_20��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp'assignvariableop_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_2_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_2_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_3_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_3_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_rmsprop_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp assignvariableop_6_rmsprop_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp(assignvariableop_7_rmsprop_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_rmsprop_momentumIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_rmsprop_rhoIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp6assignvariableop_14_rmsprop_embedding_1_embeddings_rmsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_rmsprop_dense_2_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp,assignvariableop_16_rmsprop_dense_2_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp.assignvariableop_17_rmsprop_dense_3_kernel_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp,assignvariableop_18_rmsprop_dense_3_bias_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_189
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19�
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_20"#
identity_20Identity_20:output:0*a
_input_shapesP
N: :::::::::::::::::::2$
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
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
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
�

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_24219

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Constj
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes

:2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes

:*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes

:2
dropout/GreaterEqualv
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:2
dropout/Castq
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes

:2
dropout/Mul_1\
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*
_input_shapes

::F B

_output_shapes

:
 
_user_specified_nameinputs
�
V
:__inference_global_average_pooling1d_1_layer_call_fn_24187

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_237662
PartitionedCallc
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_23817

inputs

identity_1Q
IdentityIdentityinputs*
T0*
_output_shapes

:2

Identity`

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes

:2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes

::F B

_output_shapes

:
 
_user_specified_nameinputs
�
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_24224

inputs

identity_1Q
IdentityIdentityinputs*
T0*
_output_shapes

:2

Identity`

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes

:2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes

::F B

_output_shapes

:
 
_user_specified_nameinputs
�
�
,__inference_sequential_1_layer_call_fn_24045
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_238992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*1
_input_shapes 
::::::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes

:
!
_user_specified_name	input_2"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
2
input_2'
serving_default_input_2:02
dense_3'
StatefulPartitionedCall:0tensorflow/serving/predict:�
�&
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
*X&call_and_return_all_conditional_losses
Y__call__
Z_default_save_signature"�#
_tf_keras_sequential�#{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20]}, "dtype": "float32", "input_dim": 2003, "output_dim": 7, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 20}}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 7, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20]}, "dtype": "float32", "input_dim": 2003, "output_dim": 7, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 20}}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 7, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
�

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
*[&call_and_return_all_conditional_losses
\__call__"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20]}, "dtype": "float32", "input_dim": 2003, "output_dim": 7, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
�
regularization_losses
	variables
trainable_variables
	keras_api
*]&call_and_return_all_conditional_losses
^__call__"�
_tf_keras_layer�{"class_name": "GlobalAveragePooling1D", "name": "global_average_pooling1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling1d_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 7, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7]}}
�
regularization_losses
	variables
trainable_variables
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
�

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
*c&call_and_return_all_conditional_losses
d__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7]}}
�
%iter
	&decay
'learning_rate
(momentum
)rho	rmsS	rmsT	rmsU	rmsV	 rmsW"
	optimizer
 "
trackable_list_wrapper
C
0
1
2
3
 4"
trackable_list_wrapper
C
0
1
2
3
 4"
trackable_list_wrapper
�
regularization_losses
	variables
*layer_regularization_losses
	trainable_variables
+non_trainable_variables

,layers
-metrics
.layer_metrics
Y__call__
Z_default_save_signature
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
,
eserving_default"
signature_map
):'	�2embedding_1/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
regularization_losses
	variables
/layer_regularization_losses
trainable_variables
0non_trainable_variables

1layers
2metrics
3layer_metrics
\__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
regularization_losses
	variables
4layer_regularization_losses
trainable_variables
5non_trainable_variables

6layers
7metrics
8layer_metrics
^__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
 :2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
	variables
9layer_regularization_losses
trainable_variables
:non_trainable_variables

;layers
<metrics
=layer_metrics
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
regularization_losses
	variables
>layer_regularization_losses
trainable_variables
?non_trainable_variables

@layers
Ametrics
Blayer_metrics
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 :2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
�
!regularization_losses
"	variables
Clayer_regularization_losses
#trainable_variables
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_metrics
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
H0
I1"
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
�
	Jtotal
	Kcount
L	variables
M	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	Ntotal
	Ocount
P
_fn_kwargs
Q	variables
R	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
J0
K1"
trackable_list_wrapper
-
L	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
3:1	�2"RMSprop/embedding_1/embeddings/rms
*:(2RMSprop/dense_2/kernel/rms
$:"2RMSprop/dense_2/bias/rms
*:(2RMSprop/dense_3/kernel/rms
$:"2RMSprop/dense_3/bias/rms
�2�
G__inference_sequential_1_layer_call_and_return_conditional_losses_24030
G__inference_sequential_1_layer_call_and_return_conditional_losses_24093
G__inference_sequential_1_layer_call_and_return_conditional_losses_24119
G__inference_sequential_1_layer_call_and_return_conditional_losses_24004�
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
�2�
,__inference_sequential_1_layer_call_fn_24060
,__inference_sequential_1_layer_call_fn_24134
,__inference_sequential_1_layer_call_fn_24149
,__inference_sequential_1_layer_call_fn_24045�
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
�2�
 __inference__wrapped_model_23717�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
�
input_2
�2�
F__inference_embedding_1_layer_call_and_return_conditional_losses_24158�
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
+__inference_embedding_1_layer_call_fn_24165�
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
�2�
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_24171
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_24182�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
:__inference_global_average_pooling1d_1_layer_call_fn_24187
:__inference_global_average_pooling1d_1_layer_call_fn_24176�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_2_layer_call_and_return_conditional_losses_24198�
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
'__inference_dense_2_layer_call_fn_24207�
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
�2�
D__inference_dropout_1_layer_call_and_return_conditional_losses_24224
D__inference_dropout_1_layer_call_and_return_conditional_losses_24219�
���
FullArgSpec)
args!�
jself
jinputs

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
)__inference_dropout_1_layer_call_fn_24234
)__inference_dropout_1_layer_call_fn_24229�
���
FullArgSpec)
args!�
jself
jinputs

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
�2�
B__inference_dense_3_layer_call_and_return_conditional_losses_24245�
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
'__inference_dense_3_layer_call_fn_24254�
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
�B�
#__inference_signature_wrapper_23971input_2"�
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
 ~
 __inference__wrapped_model_23717Z '�$
�
�
input_2
� "(�%
#
dense_3�
dense_3�
B__inference_dense_2_layer_call_and_return_conditional_losses_24198J&�#
�
�
inputs
� "�
�
0
� h
'__inference_dense_2_layer_call_fn_24207=&�#
�
�
inputs
� "��
B__inference_dense_3_layer_call_and_return_conditional_losses_24245J &�#
�
�
inputs
� "�
�
0
� h
'__inference_dense_3_layer_call_fn_24254= &�#
�
�
inputs
� "��
D__inference_dropout_1_layer_call_and_return_conditional_losses_24219J*�'
 �
�
inputs
p
� "�
�
0
� �
D__inference_dropout_1_layer_call_and_return_conditional_losses_24224J*�'
 �
�
inputs
p 
� "�
�
0
� j
)__inference_dropout_1_layer_call_fn_24229=*�'
 �
�
inputs
p
� "�j
)__inference_dropout_1_layer_call_fn_24234=*�'
 �
�
inputs
p 
� "��
F__inference_embedding_1_layer_call_and_return_conditional_losses_24158M&�#
�
�
inputs
� " �
�
0
� o
+__inference_embedding_1_layer_call_fn_24165@&�#
�
�
inputs
� "��
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_24171{I�F
?�<
6�3
inputs'���������������������������

 
� ".�+
$�!
0������������������
� �
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_24182N.�+
$�!
�
inputs

 
� "�
�
0
� �
:__inference_global_average_pooling1d_1_layer_call_fn_24176nI�F
?�<
6�3
inputs'���������������������������

 
� "!�������������������
:__inference_global_average_pooling1d_1_layer_call_fn_24187A.�+
$�!
�
inputs

 
� "��
G__inference_sequential_1_layer_call_and_return_conditional_losses_24004V /�,
%�"
�
input_2
p

 
� "�
�
0
� �
G__inference_sequential_1_layer_call_and_return_conditional_losses_24030V /�,
%�"
�
input_2
p 

 
� "�
�
0
� �
G__inference_sequential_1_layer_call_and_return_conditional_losses_24093U .�+
$�!
�
inputs
p

 
� "�
�
0
� �
G__inference_sequential_1_layer_call_and_return_conditional_losses_24119U .�+
$�!
�
inputs
p 

 
� "�
�
0
� y
,__inference_sequential_1_layer_call_fn_24045I /�,
%�"
�
input_2
p

 
� "�y
,__inference_sequential_1_layer_call_fn_24060I /�,
%�"
�
input_2
p 

 
� "�x
,__inference_sequential_1_layer_call_fn_24134H .�+
$�!
�
inputs
p

 
� "�x
,__inference_sequential_1_layer_call_fn_24149H .�+
$�!
�
inputs
p 

 
� "��
#__inference_signature_wrapper_23971e 2�/
� 
(�%
#
input_2�
input_2"(�%
#
dense_3�
dense_3