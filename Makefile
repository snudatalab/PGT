
define asked_delete
@read -r -p "Do you want to delete $1 [y/n] ? " answer; \
if [ $$answer = "y" ]; then \
	if [ -e $1 ]; then \
		rm -rf $1; \
	fi \
else \
	touch $1; \
	exit -1; \
fi
endef

# data_set in [adressa, glob]
DATA_SET=adressa
DATA_SET=glob

# mode in [simple, one_week, one_month, three_month]
MODE=simple
MODE=one_week
#MODE=one_month
#MODE=three_month

D2V_EMBED=default
D2V_EMBED=1000
#D2V_EMBED=300
D2V_EMBED=250

BASE_PATH=cache/$(DATA_SET)/$(MODE)
DATA_BASE_PATH=cache/$(DATA_SET)
ADRESSA_DATAS=data/simple data/one_week data/one_month data/three_month data/contentdata
GLOB_DATAS=data/glob/articles_embeddings.pickle data/glob/articles_metadata.csv data/glob/clicks

all: run

data/simple:
	$(info [Makefile] $@)

data/one_week:
	$(info [Makefile] $@)

data/three_month:
	$(info [Makefile] $@)

data/one_month:
	$(info [Makefile] $@)

data/contentdata:
	$(info [Makefile] $@)

data/glob/articles_embeddings.pickle:
	$(info [Makefile] $@)

$(DATA_BASE_PATH)/article_content.json: data/contentdata data/glob/articles_embeddings.pickle src/extract_article_content.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/extract_article_content.py -o $@ -d $(DATA_SET) -g data/glob/articles_embeddings.pickle -a data/contentdata

$(BASE_PATH)/data_per_day: $(ADRESSA_DATAS) $(GLOB_DATAS) src/raw_to_per_day.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/raw_to_per_day.py -o $@ -m $(MODE) -d $(DATA_SET)

$(BASE_PATH)/data_for_all: $(DATA_BASE_PATH)/article_content.json $(BASE_PATH)/data_per_day src/merge_days.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/merge_days.py -i $(BASE_PATH)/data_per_day -o $@ -m $(MODE) -w $(DATA_BASE_PATH)/article_content.json -d $(DATA_SET)

$(BASE_PATH)/article_info.json: data/contentdata data/glob/articles_metadata.csv $(BASE_PATH)/data_per_day src/extract_article_info.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/extract_article_info.py -u $(BASE_PATH)/data_per_day/url2id.json -o $@ -i data/contentdata -d $(DATA_SET) -g data/glob/articles_metadata.csv

$(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED): data/glob/articles_embeddings.pickle $(DATA_BASE_PATH)/article_content.json src/article_w2v.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/article_w2v.py -i $(DATA_BASE_PATH)/article_content.json -o $@ -e $(D2V_EMBED) -m cache/d2v_model/d2v_model_$(D2V_EMBED).model -d $(DATA_SET) -g data/glob/articles_embeddings.pickle

$(DATA_BASE_PATH)/glove_corpus_$(DATA_SET).pickle: $(DATA_BASE_PATH)/article_content.json src/generate_glove_corpus.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/generate_glove_corpus.py -i $(DATA_BASE_PATH)/article_content.json -o $@

$(DATA_BASE_PATH)/article_to_vec_glove.pickle_$(D2V_EMBED): $(DATA_BASE_PATH)/glove_corpus_$(DATA_SET).pickle $(DATA_BASE_PATH)/article_content.json src/article_glove.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/article_glove.py -i $(DATA_BASE_PATH)/article_content.json -o $@ -e $(D2V_EMBED) -d $(DATA_SET) -c $(DATA_BASE_PATH)/glove_corpus_$(DATA_SET).pickle

$(DATA_BASE_PATH)/url2words_glove_$(D2V_EMBED).pickle: $(DATA_BASE_PATH)/article_to_vec_glove.pickle_$(D2V_EMBED) $(DATA_BASE_PATH)/article_content.json src/generate_url2words.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/generate_url2words.py -i $(DATA_BASE_PATH)/article_content.json -o $@ -e $(D2V_EMBED) -g $(DATA_BASE_PATH)/article_to_vec_glove.pickle_$(D2V_EMBED)

$(BASE_PATH)/torch_input: $(BASE_PATH)/data_for_all src/generate_torch_rnn_input.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/generate_torch_rnn_input.py -d $(BASE_PATH)/data_for_all -o $@

d2v_rnn_torch: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/d2v_rnn_torch.py
	$(info [Makefile] $@)
	@python3 src/d2v_rnn_torch.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED)

$(BASE_PATH)/sequence_difference/$(D2V_EMBED): $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/sequence_difference.py
	$(info [Makefile] $@)
	@python3 src/sequence_difference.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) -o $@

comp_pop: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_pop.py
	$(info [Makefile] $@)
	python3 src/comp_pop.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) -w $(BASE_PATH)/pop

###################

comp_pgt: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) $(BASE_PATH)/article_info.json src/adressa_dataset.py src/comp_nert.py
	$(info [Makefile] $@)
	python3 src/comp_nert.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/nert -c $(BASE_PATH)/article_info.json

comp_pgt_wo_temp: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_nert_wo_temp.py
	$(info [Makefile] $@)
	python3 src/comp_nert_wo_temp.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/nert_wo_temp

comp_pgt_wo_attn: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_nert_wo_attn.py
	$(info [Makefile] $@)
	python3 src/comp_nert_wo_attn.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/nert_wo_attn

comp_hram: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_hram.py
	$(info [Makefile] $@)
	python3 src/comp_hram.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/hram

comp_multicell: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_multicell.py
	$(info [Makefile] $@)
	python3 src/comp_multicell.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/multicell

comp_multicell_attn: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_multicell_attn.py
	$(info [Makefile] $@)
	python3 src/comp_multicell_attn.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/multicell_attn

comp_multicell_no_dropout: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_multicell_no_dropout.py
	$(info [Makefile] $@)
	python3 src/comp_multicell_no_dropout.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/multicell_no_dropout

comp_multicell_no_attention: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_multicell_no_attention.py
	$(info [Makefile] $@)
	python3 src/comp_multicell_no_attention.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/multicell_no_attention

comp_lstm: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_lstm.py
	$(info [Makefile] $@)
	python3 src/comp_lstm.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/lstm

comp_lstm_2input: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_lstm_2input.py
	$(info [Makefile] $@)
	python3 src/comp_lstm_2input.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/lstm_2input

comp_lstm_double: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_lstm_double.py
	$(info [Makefile] $@)
	python3 src/comp_lstm_double.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/lstm_double

comp_gru4rec: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_gru4rec.py
	$(info [Makefile] $@)
	python3 src/comp_gru4rec.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/gru4rec

$(BASE_PATH)/yahoo_a2v_rnn_input.json_$(D2V_EMBED): $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) $(BASE_PATH)/article_info.json src/generate_yahoo_a2v_rnn_input.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/generate_yahoo_a2v_rnn_input.py -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -a $(BASE_PATH)/article_info.json -o $@

$(BASE_PATH)/yahoo_article2vec.json_$(D2V_EMBED): $(BASE_PATH)/yahoo_a2v_rnn_input.json_$(D2V_EMBED) $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/generate_yahoo_a2v.py
	$(info [Makefile] $@)
	$(call asked_delete, $@)
	python3 src/generate_yahoo_a2v.py -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -i $(BASE_PATH)/yahoo_a2v_rnn_input.json_$(D2V_EMBED) -w $(BASE_PATH)/yahoo_vae -o $@

comp_yahoo: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) $(BASE_PATH)/yahoo_article2vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_yahoo.py
	$(info [Makefile] $@)
	python3 src/comp_yahoo.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/yahoo -y $(BASE_PATH)/yahoo_article2vec.json

comp_yahoo_lstm: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) $(BASE_PATH)/yahoo_article2vec.json_$(D2V_EMBED) src/adressa_dataset.py src/comp_yahoo_lstm.py
	$(info [Makefile] $@)
	python3 src/comp_yahoo_lstm.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/yahoo_lstm -y $(BASE_PATH)/yahoo_article2vec.json

comp_naver: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) $(BASE_PATH)/article_info.json src/adressa_dataset.py src/comp_naver.py
	$(info [Makefile] $@)
	python3 src/comp_naver.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -c $(BASE_PATH)/article_info.json -w $(BASE_PATH)/naver

comp_npa: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) $(DATA_BASE_PATH)/url2words_glove_$(D2V_EMBED).pickle
	$(info [Makefile] $@)
	python3 src/comp_npa.py -s -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/npa -g $(DATA_BASE_PATH)/url2words_glove_$(D2V_EMBED).pickle

stat_adressa_dataset: $(BASE_PATH)/torch_input $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED) src/stat_adressa_dataset.py
	$(info [Makefile] $@)
	python3 src/stat_adressa_dataset.py -i $(BASE_PATH)/torch_input -e $(D2V_EMBED) -u $(DATA_BASE_PATH)/article_to_vec.json -w $(BASE_PATH)/stat_adress

stat_rnn_input: $(BASE_PATH)/torch_input src/stat_rnn_input.py
	$(info [Makefile] $@)
	python3 src/stat_rnn_input.py -i $(BASE_PATH)/torch_input

#run: d2v_rnn_torch
#run: comp_pop
#run: comp_lstm
#run: comp_lstm_double
#run: comp_gru4rec
#run: comp_lstm_2input
#run: comp_multicell
#run: comp_multicell_attn
#run: comp_yahoo
#run: comp_naver
#run: comp_yahoo_lstm
#run: stat_rnn_input
#run: comp_multicell_no_dropout
#run: comp_multicell_no_attention
#run: stat_adressa_dataset
#run: comp_npa
run: comp_pgt
#run: comp_pgt_wo_temp
#run: comp_pgt_wo_attn
#run: comp_hram
#run: $(DATA_BASE_PATH)/article_to_vec.json_$(D2V_EMBED)
	$(info run)


