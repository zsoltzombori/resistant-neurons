# 1. hét
Átbeszéltük a technikai dolgokat. 
Egyes neuronokat fogunk osztályozni neurális hálóval.
Ehhez kell tanítókorpusz, ezt következő hétre megcsinálom.

## Otthoni munka:
Nem determinisztikus a hasznosságszámítás, és ezzel nem tudok mit kezdeni.
Nagyon alacsony a KKI-m.
Kijovo aktivacio: zs
Bias tormokkel mi van.

# 2. hét
X_develnek megadhatom a batch sizeot
Loss és az accuracy korre
Az n+1-edik réteg aktiívációja az n-edik kimenő aktivációja

## Otthoni munka:
Bias termöket egyelőre ignorálom.
Súlyokkal baj van -> az első és utolsó rétegnél más a méret.
Az aktivációkat még meg kell regulázni.

# 3. hét
Milyen későn van a predikció, lehet, hogy nagy a szórás benne
a tanulás első felében megfigyelt hasznosság hogyan korrelál a végével 
milyen irányú a delta a nagy szórásúaknál

Ha azt látjuk, hogy a tanulás eleji és a végi hasznosság korrelál, akkor beavatkozunk
kimaszkolás mehet az elején, és hogy romlik-e a végső pontosság
vagy kevesebbel indítunk
Az iterációt le lehet venni
Az első 100 iterációnál már lehet belenyúlni
természetes baseline, hogy annyival kisebb háló meg tudja-e tanulni uganazt

fashion-mnist
!genetikus algoritmus
sorsolunk, átlagolunk, kis valószínűséggel, 

utánaolvasni, küldött Zsolt cikket slacken
bolyonganak a neuronok
dropouttal mi történik

# 5. hét 
Kétpupú eloszlás hogyan aránylik a félév eleji vonalas megfigyeléseimhez
FMNIST
Különböző osztályon belüliekre nézzem meg, meg FMNIST-en nézzem meg.

# 6.hét

Főkomponens, kör, hasznos neuron, de már a tanítás elején percentilist, vagy a végső hasznosságok percentilisét vegyem?
El kell kezdeni kísérletezni.
Talán idősoros predikció? Szerintem egész jól lehetne.
Az első 10 evaluációnál:
Ha sortolom, 0.76450 az accuracy. Csak a globális eloszlásban ÉS terjedelemben annyi infó van.
Ha nem, 0.846.

Menthetőség, idősoros predikció, amúgy is predikció. 
Mgegvan a hasznosság prediktorom ls utána újratanítok,
Meg regresszió.
Súlyoknak a statisztikája.
Prediktor.
Sok komponenssel normálissal közelítem.

Mit érdemes kipróbálni a halottakra?
- epochonként a prediktor mondja meg
- véletlen inicializálás
- visszatalál-e ugyanabba?
- átskálázom a súlyokat
- hasznos + zaj
- 2 hasznos és lineáris kombinációja + zaj
- genetikus algoritmus lépés, párokat sorsolok és azokat keresztezem, mindegyiken lesz mutaáció
- mennyire veti ez vissza a háló teljesítményét
- a kimenő és a bemenő súlyok is mozognak
- nagyon óvatosan, 1 ilyenből hogyan regenerálódik, majd több
- 

# 9. hét

-1-1-re skálázzam át, és ott prediktálgassak, standardizáljak
MSE helyett más, MAE mondjuk
A rétegtársak közti hasznosságot nézzük meg, és a prediktor rangsort sorol
Hiba az osztályozóban, milyen a megoszlása rétegeken belül

-- társadalmi mobilitás neurális hálókban
Laptopszalon, kontakt, felszámolás

# 10. hét

Esetleg unsupervised klaszterezése a neuronoknak?

Kicsit tovább pörgessem a neuronokat léptetésnél

Hogyan működik a beépített inicializáció?
Kicsi beavatkzás kell
Próbáljuk meg egy neuron
Seedeket nézzem meg.
És ha determinisztikus, akkor az elején sorsoljam újra a neuront
Train loss és dev loss

A vektorokat összefésülöm, tehát vektorkoordinátákat választok
Esetleg figyelhetek a nagyságrendbe, mindig a kisebbet/nagyobbat
A bemeneti vektorokat buzeráljam, a logit réteget hagyjam békén
Az első rétegekbe jobban belenyúlhatok, mint a végébe
Próbáljak meg más randomot, mondjuk uniformat

Tehát akkor:
koncentráljak a legrosszabbra, különböző taktikákkal
Pusztán a háló életének a leírására ez egy jó szemüveg-e.

Vizualizáció: x tengelyen neuron index, y tengelyen hasznosság két görbében, és animál!

# 11. het

Kik azok, akik az elmúlt 3 mérésen át az alsó 10 százalékban voltak
A súlyt tegyem bele a prediktorba, mind a kimenőt, mind a bemenőt.
Folytonos helyett szignumot eresszek rá, és bináris vektort kapjon

- prediktor, ami ha haszontalant prediktál, az legyen tényleg haszontalan
- binarizált vektorok, regresszió
- súlyokra koncentrálni, súlyokat bevinni, +- arány, esetleg sorsolás

# sokadik hét

1. 1000 v 10000 usefulnessű
2. periodicitás klaszterezés
3. ezek esetleg egy csoportot alkotnak
4. usefulness periodicitását prediktálgatni
5. erről a periodicitásról holnap beszélni
6. ez a hasznosság ciklus látszik-e az aktivációgörbéken
7. hasznosság vs. fonat, konfidenciagörbe, nem 0
8

- kisebb hálónál van-e ciklikusság
- idősoros predikció
- több ilyen hálón is van-e ciklikusság
- ha letakarom, eltűnik-e a ciklus
- idősor-korreláció (DTW és tsai)
- klaszterezzek periódus alapján
- periodust elkódolni az adatban (hiszen szinuszos)
- 

# Bemutató
- A dropouttal van baj, lehet attól ciklikus
- Lucidban circuiteknek hívják őket
- Lehet azzal kéne összevetni.

# január 7
- FFT, autokorr, 
- dropout nélkül (meg esetleg skálásan)
- kavarjam meg esetleg
- az aktivációkban van-e valami hullámzás, vagy sem
- Lucid Dani circuit
- ők csak empirikusan megfigyelik és nincsen semmi eszközük, amivel azonosítani tudnák őket (szemre!)
- 
HIPO:
- hasznos csoportok, amikből kilőve még elmegy a háló
HIPO2:
- a csoportok tagjainak szükségük van egymásra
- ha kilövünk egy teljes csapatot, kicsi veszteség, ha random csoportokból, nagyobb vesztesés
EGYÉB
- a gf2-n (223) több a RAM 64 GB
- a JSON kiírás soronként, valahogy 

EFOP:
- dokumentum 1 oldalban, hogy mit csináltam, milyen kísérleteket futtattam, képet is rakok bele
- 

# január 15
1. beszámoló
2. SVN-be

- 4-5 oldalas a dokumentum, de akár 1-2 oldal
- második félévben arXiv 
- neuronok hasznosssága utánaolvasás
- szélességnek köze van-e a periódushoz
- keverjem össze a train epochok elején, ez lehet, hogy statisztikailag azzal egyezik meg, hogy visszatevéses mintavétellel
- esetleg l2-vel
- képenkénti neurononkénti hasznosság
- ezeknek milyen az eloszlása
- foxámoknak rétegenkénti (mert már vannak páros gráfjaink) eoja

0. keverést nézzem meg
1. dimenziókat nézzem meg (sokkal kisebb hálón ki lehet-e mulatni)
2. periódusokat csoportosítani
3. cikkeknek utánanézni

nohup parancs & 

 # január 22

 - SVN-be töltsem fel a kért vállalást


# január 31.
- az elejével nem tudunk mit kezdeni, ott egyszerűen túl sűrűen vannak.
- meg kéne nézni, hogy mondjuk a 20 fölöttiekkel mi történik
- layerwise pretraining
  
1. bayesian háló
2. ha fontos, kicsi a szigma,
3. ha nem, akkor nagy a szigma

Nem kódolok, hanem csak olvasok, egyrészt implementáció és súlyfagyasztás, másrészt pedig bayesi háló

#############################################x
############                      ###########

# február 5. 
corpora/dep/sentences_ending_with_s_deprel.txt
Boosting: megnézi, hogy mely adaton volt magas a loss, és azt nagyobb súllyal tanítja

1. hozzak olyan implementációt, amit lehet fagyasztgatni neurononként
2. ez egyféle boosting lenne
3. random (nagy lossúakat) kiválasztok, azokat birizgálom

Valami

5. ez egy sor

# február 12.
Pytorchban implementálok.

# február 20.
A nagy aktivációjúakat lefagyasztom, miután tanult.
Utána kidobom azokat az adatokat, amiket eltalált.
És folytatom a tanulást a fagyasztott hálóval.

Kérdések:
- mennyit romlik a teljesítmény azokon, amiket már nem tanítunk
- mennyit javult azon, amit tanítunk
- milyen hatással van az általánosításra (azaz a teszthalmazra)
- hány iteráción át lehet csinálni

# február 28.

Q: Pytorch architektúra, nem feltétlen neuronok, pontos nullázás
Romlik a minden is, a háló felejt.
Gördülő átlag kéne?

- A cikkben szereplő fagyasztási módszert próbáljam ki.
- A küszöböt alacsonyabbra lehet tenni
- Kipróbálni az L2-t
- Dense-et lecserélni konvra (LeNet)
- Először fussak neki dropout/batchnorm nélkül, de azért csináljam meg

# március 6.

- CIFAR-10, CIFAR-100 
- w_2 weight decay
- batch norm
- Adrián ResNet implementáció
- megnövelném a filter számot (kimeneti csatornák számát)
- mert ezeket fogom befagyasztani
- esetleg fagyasztás és továbbtanítás után felolvasztás
- Jelasity Márk, Szeged
- softmax + crossentropy, csak néha máshogy számolják, mert numerikusan stabilabb

1. CIFAR-10, ResNet, lassabb világba lépek be
2. a kísérlet hosszabb lesz, ne aggódjak
3. Adriánnak arról is lesz véleménye, hogy mennyivel érdemes kísérletezni
4. Nézzem meg az aktivációkat
5. Csökkentsem az adathalmazt
6. 

# március 30. (most állok neki újra)

- ResNet18


# április 28.

- Miért SGD + momentum kell resnetre?
- Van a 18-nál kisebb?
- Van-e valami egyszerű, determinisztikus módja annak, hogy
- Abszolútátlag vagy sima átlag?

# május 1.

- maximumokat nézni
- töröljem ki az első kettőt (conv2, maxpool), 88.9 lesz a háló pontossága
- természetes lépés, hogy a conv5-öt leütöm
- amik együtt mozognak, azok talán jó filterek
- 50 batchsize
- gyanúsan alacsonyak a méreteim, mert a memóriába befér
- weight decay nélkül próbáljam ki
- mert a batchnorm ellátja a szerepet
- L1 reg volt a súlyokon/aktivációkon
- BN előtt, BN után, átlagos súlyok
- az egyenesek végleg feladott filterek
- kicsit túl nagy a weight decay, és akkor ugrik, amikor már nagyon szar
- A stride=2 dim csökkentő rétegek
- max 2-3-on, majd átlagolok 0-n
- batch norm és relu után max átlag
- relu: nem az, hogy mennyire nincsen orr, hanem hogy van orr, vagy nincs orr
- batch normot szoktak fontosságnak tekinteni (vagy mean, vagy stdev)
- 
- Learning efficient convolution Liu et al 2017
- Ács Judit hook

- szerintem nézzük weight decay nélkül és normális felbontású feature mapekkel, kisebb mini batch size


# május 7
- mutogassam meg 15-30 perc alat
- munkanapló


