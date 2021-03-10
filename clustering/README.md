
## clustering


### usage

`make cluster INPUT=nu`\
`nu_vectors.csv` :arrow_right: `nu_results.csv`

`make cluster INPUT=ik`\
`ik_vectors.csv` :arrow_right: `ik_results.csv`


### data

#### Hungarian postposition data

`nu_vectors.csv` = `table_4.12_vectors.csv`
= without the "English meaning" column and
values converted from the original printed table:
`0` or `0?` :arrow_right: 0;
`1` :arrow_right: 1;
`-` or `-?` :arrow_right: 2;
`?` :arrow_right: 3;
empty :arrow_right: 4

(`table_4.12_withenglish.csv` the same
 but with the _English meaning_ column)

_source:_
Table 4.12 from PhD thesis of Noémi Ligeti-Nagy

columns:
Hungarian postposition;
English meaning;
after noun?;
caseless noun?;
adjacent to noun?;
directly follows wh?;
copied to demonstrative?;
agreement on postposition?

#### Hungarian preverb data

`ik_vectors.csv` = `binary_mergedforms_vectors.csv`
= source without header, without column `Pm/Pt/Pl`\
values converted to binary: 0 if 0; 1 if >0\
:arrow_left:
because of this "szótagszám" (syllable count)
and "gyakoriság" (frequency) will be all-1,
so we will have 8 columns from which 6 are useful

_source:_
https://github.com/kagnes/prevmatrix/blob/master/actfreq_mergedforms.tsv

columns: see source header (in Hungarian)

