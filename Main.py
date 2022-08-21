import argparse
import soundfile as sf
import torch
import torchaudio
from data import Wav2Mel
import sys
import os
from os import path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from scipy.spatial.distance import cdist
from dtw import dtw, StepPattern
from scipy.spatial.distance import cdist
from heuristic_algo import STD


def speaker_conversion_demo(): 
	# Speaker style conversion 
	base_path = os.getcwd()
	model_base_path = path.join(base_path, "Model")
	model_path = path.join(model_base_path,"model-10000_17062022.ckpt")
	
	query_path = path.join(base_path, "Query")
	query1 = path.join(query_path,"query1_s1.wav")
	query2 = path.join(query_path,"query1_s2.wav")
	query3 = path.join(query_path,"query1_s3.wav")
	
		
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = torch.jit.load(model_path).to(device)
	wav2mel = Wav2Mel()

	src1, src_sr1 = torchaudio.load(query1)
	src2, src_sr2 = torchaudio.load(query2)
	src3, src_sr3 = torchaudio.load(query3)
	
	#print(src.shape)
	src1_feats = wav2mel(src1, src_sr1)[None, :].to(device) # speech doc1
	src2_feats = wav2mel(src2, src_sr2)[None, :].to(device) # speech doc2
	src3_feats = wav2mel(src3, src_sr3)[None, :].to(device) # speech doc3
	
	
	#print(src_feats.shape)
	src_1_2 = model.inference(src1_feats, src2_feats)
	bnf_1_2 = src_1_2.squeeze(0).data.numpy()
	
	src_2_2 = model.inference(src2_feats, src2_feats)
	bnf_2_2 = src_2_2.squeeze(0).data.numpy()
	
	src_3_2 = model.inference(src3_feats, src2_feats)
	bnf_3_2 = src_3_2.squeeze(0).data.numpy()
		
	z, x, y = src1_feats.shape
	s1_feats = src1_feats.cpu().detach().numpy().T
	x, y, z = s1_feats.shape
	s1_feats_flat = s1_feats.reshape(x,y).T
	
	z, x, y = src2_feats.shape
	s2_feats = src2_feats.cpu().detach().numpy().T
	x, y, z = s2_feats.shape
	s2_feats_flat = s2_feats.reshape(x,y).T
	
	z, x, y = src3_feats.shape
	s3_feats = src3_feats.cpu().detach().numpy().T
	x, y, z = s3_feats.shape
	s3_feats_flat = s3_feats.reshape(x,y).T
	
	std = STD()
	affine = std.Affinity()
	
	print(bnf_1_2.shape,bnf_2_2.shape, bnf_3_2.shape)
	sim_1_2 = affine.get_spectral_corr_sim(bnf_1_2.T, bnf_2_2.T) 
	sim_2_3 = affine.get_spectral_corr_sim(bnf_2_2.T, bnf_3_2.T) 
	
	
	matplotlib.rcParams["font.family"] = "serif"
	matplotlib.rcParams["font.size"] = "16"
	matplotlib.rcParams["axes.linewidth"] = 1 
	
	c_map = 'viridis'
	
	fig2 = plt.figure(constrained_layout=True, figsize=(19.20,10.80))
	gs = fig2.add_gridspec(3, 2, width_ratios=[5,5], height_ratios=[1,1,1], wspace=0.1)
	ax0 = fig2.add_subplot(gs[0])
	ax1 = fig2.add_subplot(gs[1])
	ax2 = fig2.add_subplot(gs[2])
	ax3 = fig2.add_subplot(gs[3])
	ax4 = fig2.add_subplot(gs[4])
	ax5 = fig2.add_subplot(gs[5])
	
	min_s_feats = 0
	max_s_feats = 80
	
	min_t_feats = 0
	max_t_feats = 80
	
	ax0.imshow(s1_feats_flat, aspect='auto', cmap=c_map) # extent=[0,400,max_s_feats,min_s_feats]
	ax2.imshow(s2_feats_flat, aspect='auto', cmap=c_map) # extent=[0,400,max_s_feats,min_s_feats],
	ax4.imshow(s3_feats_flat, aspect='auto', cmap=c_map) # extent=[0,400,max_s_feats,min_s_feats]

	ax1.imshow(bnf_1_2, aspect='auto', cmap=c_map) # extent=[0,200,max_t_feats,min_t_feats]
	ax3.imshow(bnf_2_2, aspect='auto', cmap=c_map) # extent=[0,200,max_t_feats,min_t_feats],
	ax5.imshow(bnf_3_2, aspect='auto', cmap=c_map) # extent=[0,200,max_t_feats,min_t_feats],
	
	
	ax0.invert_yaxis()
	ax1.invert_yaxis()
	ax2.invert_yaxis()
	ax3.invert_yaxis()
	ax4.invert_yaxis()
	ax5.invert_yaxis()
	
	
	ax0.set_title("(a) Speaker 1")
	ax1.set_title(r"(d)  Speaker $1 \rightarrow 2$")
	ax2.set_title("(b) Speaker 2")
	ax3.set_title("(e) Speaker 2")
	ax4.set_title("(c) Speaker 3")
	ax5.set_title(r"(f) Speaker $3 \rightarrow 2$")
	plt.savefig('speaker_style_convert.png', bbox_inches='tight')

def speaker_conversion_verify(): 
	base_path = os.getcwd()
	model_base_path = path.join(base_path, "Model")
	model_path = path.join(model_base_path,"model-10000_17062022.ckpt")
	
	query_path = path.join(base_path, "Query")
	query1 = path.join(query_path,"query1_s1.wav")
	query2 = path.join(query_path,"query1_s2.wav")
	query3 = path.join(query_path,"query1_s3.wav")
	
		
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = torch.jit.load(model_path).to(device)
	wav2mel = Wav2Mel()
	
	src1, src_sr1 = torchaudio.load(query1)
	src2, src_sr2 = torchaudio.load(query2)
	src3, src_sr3 = torchaudio.load(query3)
	
	
	src1_feats = wav2mel(src1, src_sr1)[None, :].to(device) # speech doc1
	src2_feats = wav2mel(src2, src_sr2)[None, :].to(device) # speech doc2
	src3_feats = wav2mel(src3, src_sr3)[None, :].to(device) # speech doc3
	
	
	src_1_2 = model.inference(src1_feats, src2_feats)
	bnf_1_2 = src_1_2.squeeze(0).data.numpy()
	
	src_2_2 = model.inference(src2_feats, src2_feats)
	bnf_2_2 = src_2_2.squeeze(0).data.numpy()
	
	src_3_2 = model.inference(src3_feats, src2_feats)
	bnf_3_2 = src_3_2.squeeze(0).data.numpy()
		
	z, x, y = src1_feats.shape
	s1_feats = src1_feats.cpu().detach().numpy().T
	x, y, z = s1_feats.shape
	s1_feats_flat = s1_feats.reshape(x,y).T
	
	z, x, y = src2_feats.shape
	s2_feats = src2_feats.cpu().detach().numpy().T
	x, y, z = s2_feats.shape
	s2_feats_flat = s2_feats.reshape(x,y).T
	
	z, x, y = src3_feats.shape
	s3_feats = src3_feats.cpu().detach().numpy().T
	x, y, z = s3_feats.shape
	s3_feats_flat = s3_feats.reshape(x,y).T
	
	std = STD()
	affine = std.Affinity()
	
	non_sim_1_2 = affine.get_spectral_corr_sim(s1_feats_flat.T, s2_feats_flat.T) 
	non_sim_3_2 = affine.get_spectral_corr_sim(s3_feats_flat.T, s2_feats_flat.T) 
	
	sim_1_2 = affine.get_spectral_corr_sim(bnf_1_2.T, bnf_2_2.T) 
	sim_3_2 = affine.get_spectral_corr_sim(bnf_3_2.T, bnf_2_2.T) 
	
	matplotlib.rcParams["font.family"] = "serif"
	matplotlib.rcParams["font.size"] = "16"
	matplotlib.rcParams["axes.linewidth"] = 1 
	
	c_map = 'viridis'
	
	fig2 = plt.figure(constrained_layout=True, figsize=(19.20,10.80))
	gs = fig2.add_gridspec(2, 2, width_ratios=[5,5], height_ratios=[5,5], wspace=0.1)
	ax0 = fig2.add_subplot(gs[0])
	ax1 = fig2.add_subplot(gs[1])
	ax2 = fig2.add_subplot(gs[2])
	ax3 = fig2.add_subplot(gs[3])
	
	
	min_s_feats = 0
	max_s_feats = 80
	
	min_t_feats = 0
	max_t_feats = 80
	
	im0=ax0.imshow(non_sim_1_2, interpolation='sinc', aspect='auto', cmap=c_map) # extent=[0,400,max_s_feats,min_s_feats]
	im2=ax2.imshow(sim_1_2, interpolation='sinc', aspect='auto', cmap=c_map) # extent=[0,400,max_s_feats,min_s_feats],
	
	im1=ax1.imshow(non_sim_3_2, interpolation='sinc', aspect='auto', cmap=c_map) # extent=[0,200,max_t_feats,min_t_feats]
	im3=ax3.imshow(sim_3_2, interpolation='sinc', aspect='auto', cmap=c_map) # extent=[0,200,max_t_feats,min_t_feats],
	
	ax0.invert_yaxis()
	ax1.invert_yaxis()
	ax2.invert_yaxis()
	ax3.invert_yaxis()
	
	ax0.set_title("(a)")
	ax0.set_xlabel("Speaker 2 (Mel-spec)")
	ax0.set_ylabel("Speaker 1 (Mel-spec)")
	ax1.set_title("(b)")
	ax1.set_xlabel("Speaker 2 (Mel-spec)")
	ax1.set_ylabel("Speaker 3 (Mel-spec)")
	ax2.set_title("(c)")
	ax2.set_xlabel("Speaker 2 (Mel-spec)")
	ax2.set_ylabel(r"Speaker $1\rightarrow 2$ (Mel-spec)")
	ax3.set_title("(d)")
	ax3.set_xlabel("Speaker 2 (Mel-spec)")
	ax3.set_ylabel(r"Speaker $3\rightarrow 2$ (Mel-spec)")
	
	fig2.colorbar(im0, ax=ax0, extend='both', spacing='proportional', label='correlation')
	fig2.colorbar(im1, ax=ax1, extend='both', spacing='proportional', label='correlation')
	fig2.colorbar(im2, ax=ax2, extend='both', spacing='proportional', label='correlation')
	fig2.colorbar(im3, ax=ax3, extend='both', spacing='proportional', label='correlation')
	plt.savefig('speaker_style_verify.png', bbox_inches='tight')


def heuristic_similarity_demo(): 
	base_path = os.getcwd()
	model_base_path = path.join(base_path, "Model")
	model_path = path.join(model_base_path,"model-10000_17062022.ckpt")
	
	query_file = path.join(base_path,"Query","query2_s4.wav")
	doc_path = path.join(base_path,"Corpus","doc2.wav")	
	
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = torch.jit.load(model_path).to(device)
	#print(model)
	wav2mel = Wav2Mel()
	
	src, src_sr = torchaudio.load(doc_path)
	tgt, tgt_sr = torchaudio.load(query_file)

	src = wav2mel(src, src_sr)[None, :].to(device) # speech doc
	tgt = wav2mel(tgt, tgt_sr)[None, :].to(device) # query
	
	cvt = model.inference(tgt, src) # note that we are normalising the keyword query w.r.to src doc 
	cvt_qry_norm = cvt.squeeze(0).data.T.numpy()
	
	print(cvt_qry_norm.shape)
	
	cvt_doc = model.inference(src, src)
	cvt_doc_norm = cvt_doc.squeeze(0).data.T.numpy()
	
	print(cvt_doc_norm.shape)
	
	std = STD()
	affine = std.Affinity()
	
	bnf_qry = cvt_qry_norm
	bnf_doc = cvt_doc_norm
	
	hybrid_mat, h_cost, st, end = affine.compute_heuristic_affinity(bnf_qry, bnf_doc)
	
	hybrid_mat = np.where(hybrid_mat>=0.8,hybrid_mat,0)
	
	c_map = 'Greys'
	
	matplotlib.rcParams["font.family"] = "serif"
	matplotlib.rcParams["font.size"] = "16"
	matplotlib.rcParams["axes.linewidth"] = 1 
	
	
	fig3 = plt.figure(constrained_layout=True, figsize=(19.20,10.80))
	gs = fig3.add_gridspec(2,1)
	ax00 = fig3.add_subplot(gs[0])
	ax01 = fig3.add_subplot(gs[1], sharex=ax00)
	im1 = ax00.imshow(hybrid_mat.T, aspect='auto', interpolation='nearest', cmap=c_map)
	

	x_stem = [i+1 for i in range(len(h_cost))]
	
	x_val = [i+1 for i in range(hybrid_mat.shape[0])]
	ax01.plot(x_val, h_cost, linewidth=2, color='b')
	ax01.axvline(st+2, color='r', linestyle='--', linewidth=3)
	ax01.axvline(end+1, color='r', linestyle='--', linewidth=3)
	
	ax00.axvline(st+2, color='r', linestyle='--', linewidth=3)
	ax00.axvline(end+1, color='r', linestyle='--', linewidth=3)
	
	ax00.set_ylabel("Query (frames)")
	ax01.set_ylabel(r"$H_{sim}$")
	ax00.set_title("(a) Similarity matrix")
	ax01.set_title("(b) Heuristic Similarity")
	ax01.set_xlabel("Document (frames)")
	ax00.set_rasterized(True)
	ax01.set_rasterized(True)
	plt.savefig('heuristic_similarity_match.png', bbox_inches='tight')
	print("Plot saved successfully")


if __name__=="__main__": 
	# step 1 --> compute the speaker style conversion 
	speaker_conversion_demo()
	
	# step 2 --> verify the speaker style conversion 
	speaker_conversion_verify()
	
	# step 3 --> Capture the heuristic similarity
	heuristic_similarity_demo()
