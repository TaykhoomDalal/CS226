q3.data

Each row is an individual.
Column 'individual' provides a unique identifier for each individual.
Column 'population' is the true population label for the individual.
Columns that start with 'rs' are SNPs. 

Sample code to read in the data:

	# install.packages('tidyverse') # uncomment to install tidyverse
	library('tidyverse')
	data = read_tsv('q1.data')
	snps = data %>% select(starts_with('rs'))

Sample code for plotting:

	# pca_df should contain your principal components obtained 
	# from the prcomp function

	ggplot(data = pca_df) + 
	  geom_point(aes(x = PC1, y = PC2, color = population)) + 
	  scale_color_brewer(palette ='Set1') +
	  theme_bw() 
