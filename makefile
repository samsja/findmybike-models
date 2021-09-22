tests:
	pytest tests

formatting:
	black lbc_scrapper/
	isort lbc_scrapper/
	black captcha_pass_ia/
	isort captcha_pass_ia/

notebook-sync:
	jupytext --sync  notebooks/*.ipynb


clean_log:
	rm -rf lightning_logs

tensorboard:
	tensorboard --logdir ./lightning_logs
