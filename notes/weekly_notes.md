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

