import csv
import json
import sys, os
from collections import defaultdict, Counter
import numpy as np

min_size=4
min_freq = 3
skip = 100

##################################################
##Read the wordlist and do some basic filtering###
##################################################
if os.path.exists("D.json"):
	with open("D.json", 'r') as infile: D = json.loads( infile.read() )
else:
	with open('SUBTLEXusExcel2007.csv', 'r') as infile:
		csvreader = csv.reader(infile)
		head = next(csvreader)
		D = {}
		for i,row in enumerate(csvreader):
			if row[0][0].isupper() or (int(row[1]) < min_freq) or (len(row[0]) < min_size) or (i < skip): continue
			D[ row[0].lower() ] = int(row[1]) 
	with open("D.json", 'w') as outfile: outfile.write( json.dumps(D) )

most_frequent = max( D.values() )


##################################################
##Get the most common words					   ###
##################################################
top = 150
Dtop = { k:v for k,v in D.items() if v >= top}
print( r'|D({})| ='.format(top), len(Dtop) )

##################################################
##Clean the IPA dictionary 					   ###
##################################################
if os.path.exists("homonyms.json"):
	with open("homonyms.json", 'r') as infile: H = json.loads( infile.read() )
else:
		
	allowed = {'ɛ', 'ʊ', 'ɹ', 'o', 'u', 'ɒ', 'j', 's', 'f', 'ʌ', 'h', 'ɜ', 'ɪ', 'p', 'ʃ', 'θ', 'ŋ', 'd', 'e', 'x', 't', 'l', 'n', 'v', 'ɔ', 'm', 'ɫ', 'ə', 'r', 'w', 'ð', 'z', 'ɡ', 'i', 'ʒ', 'ɝ', 'b', 'ɬ', 'ɑ', 'æ', 'ɐ', 'a', 'k'}

	IPA = defaultdict( set )
	chars = set()
	for name in ['en_UK.txt', 'en_US.txt']:
		with open(name, 'r') as infile:
			for line in infile:
				w,p = line.split("\t")
				w = w.strip()
				p = p.strip()
				
				for pr in p.split(','):
					IPA[w].add(  ''.join([c for c in pr if c in allowed])  )
					for c in pr: chars.add(c)


	for w in IPA: IPA[w] = list(IPA[w])

	common_words = []
	H = {}

	for w in D:
		if w in IPA: 
			common_words.append(w)
		else:
			H[w] = []

		
	nh = 0
	for i,w in enumerate(common_words): #for every word
		
		if i>0 and i%100 == 0: 
			print(i,"/", len(common_words), "found", nh, "homonyms")

		hom = []	
		for pw in IPA[w]: #for every pronounciation of that word
			
			for w2 in common_words: #check every other word
				if pw in IPA[w2] and w2 != w: #if w2 sounds like w
					hom.append(w2) #w2 is a homonym
		
		if len(hom): nh += 1
		H[w] = list(set(hom))
			
	with open("homonyms.json", 'w') as outfile: outfile.write( json.dumps(H) )

print( sum( int(len(H[w])>0) for w in H ), "words have homonyms in D({})".format(min_freq) ) 

##################################################
###Generate confusion sets for ever word       ###
##################################################
if os.path.exists("confusion.json"):
	with open("confusion.json", 'r' ) as infile: C = json.loads( infile.read() )
else:
	
	alphabet = list('abcdefghijklmnopqrstuvwxyz') + ['']
	C = {}
	
	for i,w1 in enumerate(D):

		##Above is too slow, so construct the confusions myself	
		confusions = set()
		for j in range(len(w1)):
			if j < len(w1)-1:
				transp = w1[:j] + w1[j+1] + w1[j] + w1[j+2:]	##flip adjacent letters
				if transp in Dtop: confusions.add(transp)
			for c in alphabet:
				ins = w1[:j] + c + w1[j:]	
				if ins in Dtop: confusions.add(ins)
				sub = w1[:j] + c + w1[j+1:] #substitution of empty string = deletion
				if sub in Dtop: confusions.add(sub)
		##tail insertion
		for c in alphabet:
			ins = w1 + c
			if ins in Dtop: confusions.add(ins)		
			
		#homonyms
		confusions = confusions.union( set([ w2 for w2 in H[w1] if w2 in Dtop ]) )
			
		#if you don't want original word on the list
		#if w1 in confusions: confusions.remove(w1)
		#but you usually do
		confusions.add(w1)
	
	
		C[w1] = list(confusions)
	
		if i>0 and i % 1000 == 0: 
			print(i,"/", len(D), "confusions sets generated")


	with open("confusion.json", 'w' ) as outfile: outfile.write(json.dumps(C))

print( sum( int(len(C[w])>1) for w in C ), "words have non-trivial confusion sets" ) 

##################################################
###Plot the global confusion graph			   ###
##################################################
import matplotlib.pyplot as plt
from scipy.stats import poisson
plt.rcParams.update({'font.size': 18})
Dlist = np.array(list(D.keys()))[ np.argsort( -np.array( list(D.values()) ) ) ] #sorted by frequency, highest to lowest


fig, ax = plt.subplots( 1,2, figsize=(20,10) )
counts = [ len(v)-1 for k,v in C.items() ]
mu = np.mean(counts)
ax[0].hist(counts, bins=range(15), density=True )
x = np.arange(11)+0.5
ax[0].plot(x, poisson.pmf(x, mu), label=r'$\lambda_1 = {:.2f}$'.format(mu))
ax[0].set_xlabel(r"$c$")
ax[0].set_ylabel(r"$p(c)$")
ax[0].legend(loc="upper right")


np.random.seed(12345)
Ns = 1000000
counts = []
sum_counts = []
print("Generating Global Confusion...")
for s in range(Ns):
	ids = np.random.randint(0,len(Dlist),3)
	sum_counts.append( sum( [len(C[ Dlist[k] ])-1 for k in ids] ) )	 

	count = 1;
	for k in ids: count *= len(C[ Dlist[k] ])	
	count -= 1
	
	counts.append(count )

mu = np.mean(counts)
freq, bins = np.histogram(counts, bins=range(max(counts)+1), density=True)
print("lambda_3 =", mu, "p_3(0)=",freq[0], "p_3(c>4)", sum(freq[4:]) )
ax[1].hist(counts, bins=range(20), density=True, label=r'$c_\Pi$: $p_3(0) = {:.2f}$, '.format(freq[0]) + r'$p_3(c_\Pi>3) = {:.2f}$, '.format( sum(freq[4:]) ) + r'$\lambda_3 = {:.2f}$'.format(mu), alpha=0.25)
mu = np.mean(sum_counts)
freq, bins = np.histogram(sum_counts, bins=range(max(sum_counts)+1), density=True)
print("lambda_3 =", mu, "p_3(0)=",freq[0], "p_3(c>4)", sum(freq[4:]) )
ax[1].hist(sum_counts, bins=range(20), density=True, label=r'$c_\Sigma$: $p_3(0) = {:.2f}$, '.format(freq[0]) + r'$p_3(c_\Sigma>3) = {:.2f}$, '.format( sum(freq[4:]) ) + r'$\lambda_3 = {:.2f}$'.format(mu), color='C2', alpha=0.25)
ax[1].set_xlabel(r"$c$")
ax[1].set_ylabel(r"$p_3(c)$")
ax[1].legend(loc="upper right")

ax[0].set_title("(a)")
ax[1].set_title("(b)")
plt.savefig("global_confusion.png", dpi=fig.dpi)
#plt.show()
plt.close()



##################################################
### W3W conversion functions				   ###
##################################################


def lat_lon_to_XYxyW(lat, lon):
	X = np.floor( (lon+180) * 24 )
	Y = np.floor( (lat+90) * 24 )
	W = np.maximum(1,np.floor(1546*np.cos( ((Y+0.5)/24-90)* np.pi/180 )).astype(np.int64) )
	x = np.floor(W * np.modf((lon+180) * 24)[0] )
	y = np.floor(1546*np.modf((lat+90) * 24)[0] )
	return X.astype(np.int64),Y.astype(np.int64), x.astype(np.int64), y.astype(np.int64), W.astype(np.int64)

def XYxy_to_lat_lon(X,Y,x,y):
	W = np.maximum(1,np.floor(1546*np.cos( ((Y+0.5)/24-90)* np.pi/180 )).astype(np.int64) )

	return (Y + ((y+0.5)/1546))/24 - 90, (X + ((x+0.5)/W))/24 - 180
	
		
def xy_to_n(x,y, q=0):
	return q + 1546*x + y
	
def n_to_m(n, b=20000000000, a=3639313, c=0):
	return c +(	a*n % b ) 

def m_to_tuple(m):
	l = math.floor( m**(1/3) )
	lp = [l**i for i in range(4)]
	if m < lp[3] + lp[2] + 2*l + 1:
		r = m-lp[3]
		return l,r//(l+1),r%(l+1)
	elif m < lp[3] + 2*lp[2] + 3*l + 1:
		r = m-(lp[3] + lp[2] + 2*l+1)
		return r//(l+1),l,r%(l+1)
		
	r = m-(lp[3] + 2*lp[2] + 3*l+1)
	return (r//l, r%l, l)
	
##################################################
###Plot the global confusion graph			   ###
##################################################
from geopy import distance
from scipy.sparse import coo_matrix
import math

#use indices
word_to_id = { w:i for i,w in enumerate(Dlist) }
id_to_word = { i:w for i,w in enumerate(Dlist) }
Cid = {}
for w in C:
	wid = word_to_id[w]
	Cid[wid] = [ word_to_id[c] for c in C[w] ]


address_book = defaultdict(lambda: defaultdict(list)) #{123:{456:[7,8,9]}}
test = []
address_map = {}

V = 1546*961
X = 4316 
Y = 3396

slat, slon = XYxy_to_lat_lon(X,Y,0,0)
flat, flon = XYxy_to_lat_lon(X+3,Y,961,1546)
print( "Simulate W3W...")
print( "start lat, lon:", slat, slon )
print( "final lat, lon:", flat, flon )
print( "X distance =", distance.distance( (slat,slon) , (slat,flon) ).km , "km")
print( "Y distance =", distance.distance( (slat,slon) , (flat,slon) ).km , "km")

if os.path.exists("address_book.json"):
	with open("address_book.json", 'r' ) as infile: address_book = json.loads( infile.read() )
	with open("address_map.json", 'r' ) as infile: address_map = json.loads( infile.read() )
else:


			
	for qi in range(3):
		q = 1 + qi*V
		print("Cell ::", qi )

		for x in range(961): 
			for y in range(1546): 
							
				n = xy_to_n(x,y,q=q)
				m = n_to_m(n)
				i,j,k = m_to_tuple(m) 

				address_book[ id_to_word[i] ][ id_to_word[j] ].append( id_to_word[k] )
				address_map[ ".".join([id_to_word[c] for c in (i,j,k)]) ] = [ XYxy_to_lat_lon(X,Y,x,y), (961*qi + x,y) ]
				
		X += 1

	with open("address_book.json", 'w' ) as outfile: outfile.write(json.dumps(address_book))
	with open("address_map.json", 'w' ) as outfile: outfile.write(json.dumps(address_map))


cpairs = 0;
tree_confusions = defaultdict(list)


for i in address_book:
	for j in address_book[i]:
		for k in address_book[i][j]:		
		
			for i2 in C[i]:
				if i2 not in address_book: continue
				for j2 in C[j]:
					if j2 not in address_book[i2]: continue
					for c in address_book[i2][j2]:
						if c in C[k] and (i,j,k) != (i2,j2,c):
							tree_confusions[ ".".join([i,j,k]) ].append( ".".join([i2,j2,c]) )
							cpairs += 1


print("# confusable addresses =", cpairs ) #cpairs = 1300 in 1
									#cpairs = 12158 in 3

unique_pairs = set()
for k in tree_confusions:
	for v in tree_confusions[k]:
		unique_pairs.add( tuple(sorted( [k,v] )) )

print("# unique pairs",len(unique_pairs))


dists = []
N = 961*3 	
M = 1546
row = []
col = []
data = []

for k,v in unique_pairs:
	d = distance.distance( address_map[k][0], address_map[v][0] ).km
	x1,y1 = address_map[k][1]
	x2,y2 = address_map[v][1]
	dists.append( d )
	row.append( x1 )
	row.append( x2 )
	col.append( y1 )
	col.append( y2 )
	ne = sum([ int(i==j) for i,j in zip(k.split("."),v.split(".")) ])
	data.append( ne )
	data.append( ne )
	

import matplotlib.gridspec as gridspec
fig = plt.figure(tight_layout=True, figsize=(24,8))
gs = gridspec.GridSpec(1, 3)

ax0 = fig.add_subplot(gs[0])
ax0.hist(dists, bins=30)
ax0.set_xlabel("Distance (km)")
ax0.set_ylabel("Number of Confusable Pairs")
ax0.set_title("(a)")


sp = Counter(data)
print( "Number of shared words", sp ) #Counter({1: 688, 0: 324, 2: 288})
ax1 = fig.add_subplot(gs[1:])
row = np.array(row)
col = np.array(col)
data = np.array(data)
colors = ['y','c','m']
for i in range(3):
	
	idx = np.array( [ j for j in range(len(data)) if data[j]==i ] )
	ax1.scatter(row[idx], col[idx], marker='s', c=colors[i], label=r"{} confusable pairs with {} identical words".format( int(sp[i]/2), i), s=3)

ax1.axvline(x=961, linestyle="--", color='k')
ax1.axvline(x=961*2, linestyle="--", color='k')
ax1.set_xlim([-1,N])
ax1.set_ylim([-1,M])
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("(b)")
lgnd = ax1.legend(markerscale=6, loc="upper left")


plt.savefig("local_confusion.png", dpi=fig.dpi)
plt.show()
plt.close()

