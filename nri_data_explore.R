# create maps of land use shares from nri data
# update for github

library(tidyverse)
library(sf)
library(tmap)

lu <- tbl_df(readRDS('landuse_shares.rds'))
lu$fips <- factor(str_pad(lu$fips, 5, side = 'left', pad = '0'))
counties <- st_read('Census Boundaries/conus_county.shp')

d1 <- counties %>%
	left_join(lu, by = c('GEOID'='fips'))


######################################################################
# 
######################################################################
frmap_south <- tm_shape(filter(d1, region_name=='South')) +
	tm_fill('fr_acres_change', palette = "RdYlGn", style = 'quantile', n=5, title = 'Change in Forest Acres', legend.hist = F)+
	tm_layout(scale=0.6, legend.position = c('right','bottom'))+
	tm_shape(filter(d1, region_name=='South'))+
	tm_borders(lwd=.5, alpha = .5)

agmap <- tm_shape(d1) +
	tm_fill('ag_acres_change', palette = "PRGn", style = 'quantile', n=7, title = 'Change in Ag Acres', legend.hist = T)+
	tm_layout(scale=0.75, legend.position = c('left','bottom'))+
	tm_shape(d1)+
	tm_borders(lwd=.5, alpha = .5)

frmap_north <- tm_shape(filter(d1, region_name=='North')) +
	tm_fill('fr_acres_change', palette = "PRGn", style = 'quantile', n=5, title = 'Change in Forest Acres', legend.hist = F)+
	tm_layout(scale=0.6, legend.position = c('right','bottom'))+
	tm_shape(filter(d1, region_name=='North'))+
	tm_borders(lwd=.5, alpha = .5)

agmap <- tm_shape(d1) +
	tm_fill('ag_acres_change', palette = "PRGn", style = 'quantile', n=7, title = 'Change in Ag Acres', legend.hist = T)+
	tm_layout(scale=0.75, legend.position = c('left','bottom'))+
	tm_shape(d1)+
	tm_borders(lwd=.5, alpha = .5)

urmap <- tm_shape(d1) +
	tm_fill('ur_acres_change', palette = "PRGn", style = 'quantile', n=7, title = 'Change in Urban Acres', legend.hist = T)+
	tm_layout(scale=0.75, legend.position = c('left','bottom'))+
	tm_shape(d1)+
	tm_borders(lwd=.5, alpha = .5)

tmap_arrange(frmap, agmap, urmap, asp = NA)

######################################################################
######################################################################
frmap2 <- tm_shape(d1) +
	tm_fill('fr_share_change', palette = "PRGn", style = 'quantile', n=7, title = 'Change in Forest Share', legend.hist = T)+
	tm_layout(scale=0.75, legend.position = c('left','bottom'))+
	tm_shape(d1)+
	tm_borders(lwd=.5, alpha = .5)

agmap2 <- tm_shape(d1) +
	tm_fill('ag_share_change', palette = "PRGn", style = 'quantile', n=7, title = 'Change in Ag Share', legend.hist = T)+
	tm_layout(scale=0.75, legend.position = c('left','bottom'))+
	tm_shape(d1)+
	tm_borders(lwd=.5, alpha = .5)

urmap2 <- tm_shape(d1) +
	tm_fill('ur_share_change', palette = "PRGn", style = 'quantile', n=7, title = 'Change in Urban Share', legend.hist = T)+
	tm_layout(scale=0.75, legend.position = c('left','bottom'))+
	tm_shape(d1)+
	tm_borders(lwd=.5, alpha = .5)

tmap_arrange(frmap2, agmap2, urmap2, asp = NA)
